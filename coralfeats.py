import caffe
import boto
import os
import json

import numpy as np

import beijbom_vision_lib.caffe.tools as bct
import coral_lib.patch.tools as cpt

from boto.s3.key import Key
from boto.sqs.message import Message




def coralfeats():
    print 'entering coralfeats'
    caffe.set_mode_cpu()
    net = caffe.Net('deploy.prototxt', 'VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)
    qconn = boto.sqs.connect_to_region("us-west-2", aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)
    q = qconn.get_queue('spacerjobs')

    m = q.read()
    bodydict = json.loads(m.get_body())
    print bodydict


    s3conn = boto.connect_s3(aws_access_key_id, aws_secret_access_key)
    bucket = s3conn.get_bucket(bodydict['bucketname'], validate=True)

    key = bucket.get_key(bodydict['imkey'])
    key.get_contents_to_filename(bodydict['imkey'])

    pyparams = {
	'im_mean': [128, 128, 128],
        'scaling_method':'scale',
        'scaling_factor':1,
        'crop_size':224,
        'batch_size':10}
    imlist = [bodydict['imkey']]
    imdict = dict()
    imdict[bodydict['imkey']] = ([], 100)
    for row, col in bodydict['rowcols']:
       imdict[bodydict['imkey']][0].append((row, col, 1)) 
    
    (_, _, feats) = cpt.classify_from_patchlist(imlist, imdict, pyparams, net, scorelayer = 'fc7')    
        
    print len(feats), len(feats[0])
    print "dockermount!"
coralfeats()
