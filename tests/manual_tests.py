import sys
sys.path.append('/root/caffe')

import caffe
import boto
import os
import json

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
