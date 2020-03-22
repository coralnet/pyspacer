"""
Defines the highest-level method for task handling through AWS SQS.
"""

import json

from spacer import config
from spacer.messages import TaskMsg, TaskReturnMsg
from spacer.tasks import extract_features, train_classifier, classify_features


def sqs_mailman(queue_group='spacer') -> bool:
    print("-> Grabbing message.")

    # Load default queue
    conn = config.get_sqs_conn()
    inqueue = conn.get_queue('{}_jobs'.format(queue_group))
    resqueue = conn.get_queue('{}_results'.format(queue_group))

    # Read message
    m = inqueue.read()
    if m is None:
        print("-> No messages in inqueue.")
        return False
    body = m.get_body()

    # Try to deserialize message
    try:
        task_msg_dict = json.loads(body)
        task_msg = TaskMsg.deserialize(task_msg_dict)
        task_return_msg = process_task(task_msg)
        return_msg_dict = task_return_msg.serialize()
    except Exception as e:
        # Handle deserialization errors directly in mailman.
        # All other errors are handled in "handle_message" function.
        return_msg_dict = {
            'original_job': body,
            'ok': False,
            'results': None,
            'error_message': 'Error deserializing message: ' + repr(e)
        }

    # Return
    m_out = resqueue.new_message(body=json.dumps(return_msg_dict))
    resqueue.write(m_out)
    inqueue.delete_message(m)
    return True


def process_task(task_msg: TaskMsg) -> TaskReturnMsg:

    task_defs = {
        'extract_features': extract_features,
        'train_classifier': train_classifier,
        'classify_features': classify_features,
    }

    assert isinstance(task_msg, TaskMsg)
    assert task_msg.task in config.TASKS

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
    sqs_mailman()  # pragma: no cover
