import boto
import json

from tasks import extract_features, train_robot, classify_image
from boto.sqs.message import Message

tasks = {
    'extract_features': extract_features,
    'train_robot': train_robot,
    'classify_image': classify_image
}

def grab_message():
    print "grabbing message."

    # Load default queue
    conn = boto.sqs.connect_to_region("us-west-2")
    inqueue = conn.get_queue('spacer_jobs')
    resqueue = conn.get_queue('spacer_results')
    errorqueue = conn.get_queue('spacer_errors')

    # Read message
    m = inqueue.read()
    if m is None:
    	print "No messages in inqueue."
    	return 1
    body = json.loads(m.get_body())
    
    # Do the work
    try:
        status, outbound = handle_message(body)
        inqueue.delete_message(m)
        
        m = Message()
        out_body = {'inbound': body, 'outbound': outbound}
        m.set_body(json.dumps(out_body))
        resqueue.write(m)

    except Exception as e:
    	print e
        m = Message()
        m.set_body(json.dumps(e.message))
        errorqueue.write(m)
    

def handle_message(body):
    
    if not type(body) is dict:
        raise TypeError('Input "body" must be a dictionary.')

    if not 'task' in body:
        raise KeyError('Input dictinary "body" must have key "task"')

    if not body['task'] in tasks:
        raise ValueError('Requested task: "{}" is not a valid task'.format(body['task']))
    
    result = tasks[body['task']](body['payload'])

    resbody = {
        'inbound':body,
        'result':result
    }

    return 1, resbody

if __name__ == '__main__':
    grab_message()
