import boto
from tasks import extract_features, train_robot, classify_image


tasks = {
	'extract_features': extract_features,
	'train_robot': train_robot,
	'classify_image': classify_image
}


def pickup_message():

	# Load default queue
	queue = boto.sqs.connect_to_region("us-west-2").get_queue('spacerjobs')

	# Read message
    m = q.read()
    bodydict = json.loads(m.get_body())
    print bodydict

    if m['task'] in tasks:
    	tasks[m['task']](m['payload'])
    else:
    	raise Exception('Task {} not implemented'.format(m['task']))



 

