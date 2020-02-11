import json

import boto
from boto.sqs.message import Message

from spacer.messages import TaskMsg
from spacer.tasks import extract_features, train_classifier, deploy

tasks = {
    'extract_features': extract_features,
    'train_classifier': train_classifier,
    'deploy': deploy
}


def grab_message(queue_group='spacer'):
    print("grabbing message.")

    # Load default queue
    conn = boto.sqs.connect_to_region("us-west-2")
    inqueue = conn.get_queue('{}_jobs'.format(queue_group))
    resqueue = conn.get_queue('{}_results'.format(queue_group))

    # Read message
    m = inqueue.read()
    if m is None:
        print("No messages in inqueue.")
        return 1
    body = json.loads(m.get_body())

    task_msg = TaskMsg(body['task'], body['payload'])

    out_body = handle_message(task_msg)
    
    m_out = Message()
    m_out.set_body(json.dumps(out_body))
    resqueue.write(m_out)
    inqueue.delete_message(m)
    

def handle_message(task_msg):

    if task_msg.task not in tasks:
        raise ValueError('Requested task: "{}" is not a valid task'.
                         format(task_msg.task))

    try:
        out_body = {
            'original_job': '',
            'ok': True,
            'result': tasks[task_msg.task](task_msg.payload),
            'error_message': None
        }
    except Exception as e:
        out_body = {
            'original_job': '',
            'ok': False,
            'result': None,
            'error_message': repr(e)}

    return out_body


if __name__ == '__main__':
    grab_message()
