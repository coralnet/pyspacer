import json

from typing import Dict

import boto
from boto.sqs.message import Message

from spacer.messages import TaskMsg, TaskReturnMsg
from spacer.tasks import extract_features_task, train_classifier_task, deploy


def grab_message(queue_group='spacer'):
    print("-> Grabbing message.")

    # Load default queue
    conn = boto.sqs.connect_to_region("us-west-2")
    inqueue = conn.get_queue('{}_jobs'.format(queue_group))
    resqueue = conn.get_queue('{}_results'.format(queue_group))

    # Read message
    m = inqueue.read()
    if m is None:
        print("-> No messages in inqueue.")
        return 1
    task_msg_dict = json.loads(m.get_body())

    # Try to deserialize message
    try:
        task_msg = TaskMsg.deserialize(task_msg_dict)
        task_return_msg = handle_message(task_msg)
        return_msg_dict = task_return_msg.serialize()
    except Exception as e:
        # Handle deserialization errors directly in mailman.
        # All other errors are handled in "handle_message" function.
        return_msg_dict = {
            'original_job': task_msg_dict,
            'ok': False,
            'results': None,
            'error_message': repr(e)
        }

    # Return
    m_out = Message()
    m_out.set_body(json.dumps(return_msg_dict))
    resqueue.write(m_out)
    inqueue.delete_message(m)
    

def handle_message(task_msg: TaskMsg) -> TaskReturnMsg:

    task_defs = {
        'extract_features': extract_features_task,
        'train_classifier': train_classifier_task,
        'deploy': deploy
    }

    try:
        return_msg = TaskReturnMsg(
            original_job=task_msg,
            ok=True,
            results=task_defs[task_msg.task](task_msg.payload),
            error_message=None
        )
    except Exception as e:
        return_msg = TaskReturnMsg(
            original_job=task_msg,
            ok=False,
            results=None,
            error_message=repr(e)
        )
    return return_msg


if __name__ == '__main__':
    grab_message()
