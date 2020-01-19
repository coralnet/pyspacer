import boto
import json

from tasks import extract_features, train_classifier, deploy
from boto.sqs.message import Message

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
    errorqueue = conn.get_queue('{}_errors'.format(queue_group))

    # Read message
    m = inqueue.read()
    if m is None:
        print("No messages in inqueue.")
        return 1
    body = json.loads(m.get_body())
    
    # Do the work
    try:
        outbound = handle_message(body)
        out_body = {'original_job': body, 'result': outbound}
        queue = resqueue
        
    except Exception as e:
        out_body = {'original_job': body, 'error_message': e.message}
        queue = errorqueue
    
    m_out = Message()
    m_out.set_body(json.dumps(out_body))
    queue.write(m_out)
    inqueue.delete_message(m)
    

def handle_message(body):
    
    if not type(body) is dict:
        raise TypeError('Input "body" must be a dictionary.')

    if 'task' not in body:
        raise KeyError('Input dictionary "body" must have key "task"')

    if not body['task'] in tasks:
        raise ValueError('Requested task: "{}" is not a valid task'.format(body['task']))
    
    return tasks[body['task']](body['payload'])


if __name__ == '__main__':
    grab_message()
