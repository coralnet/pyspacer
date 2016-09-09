import sys
sys.path.append('/root/caffe')
sys.path.append('../')

import caffe
import boto
import os
import json
import argparse
from spacer import mailman

import numpy as np

import beijbom_vision_lib.caffe.tools as bct
import coral_lib.patch.tools as cpt

from boto.s3.key import Key
from boto.sqs.message import Message

import tasks

with open('secrets.json') as data_file:    
    data = json.load(data_file)
    os.environ['AWS_ACCESS_KEY_ID'] = data['aws_access_key_id']
    os.environ['AWS_SECRET_ACCESS_KEY'] = data['aws_secret_access_key']

### Some code to copy
def test_feats():
	payload = {
		'bucketname': 'spacertestbucket',
		'imkey': 'dog.jpeg',
		'outputkey': 'dog.jpeg.feats',
		'modelname': 'vgg16_coralnet_ver1',
		'rowcols': [(100, 140), (100, 240)]
	}
	feats = tasks.extract_features(payload)
	
	return feats

def read_messages(qname):
	print 'reading 1 message from {}'.format(qname)
	q = boto.sqs.connect_to_region("us-west-2").get_queue(qname)
	m = q.read()
	print m.get_body()

def run_mailman(sdf):
	mailman.grab_message()


parser = argparse.ArgumentParser(description='Manual test function.')
parser.add_argument('task', type=str)
parser.add_argument('queue', type=str)
dispatcher={
	'run_mailman':run_mailman,
	'read_messages':read_messages
}
args = parser.parse_args()
dispatcher[args.task](args.queue)
