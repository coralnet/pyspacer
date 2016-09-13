import sys
sys.path.append('/root/caffe/')

import caffe
import os
import boto
import json
import time

import beijbom_vision_lib.caffe.tools as bct
import coral_lib.patch.tools as cpt

from boto.s3.key import Key

def extract_features(payload):
    print "Extracting features for {}.".format(payload['imkey'])
    t1 = time.time()

    # Make sure the right model and prototxt are available locally.
    was_cashed = _download_models(payload['modelname'])

    # Download the image to be processed.
    conn = boto.connect_s3()
    bucket = conn.get_bucket(payload['bucketname'], validate=True)

    key = bucket.get_key(payload['imkey'])
    basename = os.path.basename(payload['imkey'])
    key.get_contents_to_filename(basename)

    # Setup caffe
    caffe.set_mode_cpu()
    net = caffe.Net('../models/' + str(payload['modelname'] + '.deploy.prototxt'), '../models/' + str(payload['modelname'] + '.caffemodel'), caffe.TEST)
    
    # Set parameters
    pyparams = {
    'im_mean': [128, 128, 128],
        'scaling_method':'scale',
        'scaling_factor':1,
        'crop_size':224,
        'batch_size':10}
    
    imlist = [basename]
    imdict = {
        basename:([], 100)
    }
    for row, col in payload['rowcols']:
       imdict[basename][0].append((row, col, 1)) 

    # Run
    t2 = time.time()
    (_, _, feats) = cpt.classify_from_patchlist(imlist, imdict, pyparams, net, scorelayer = 'fc7')
    feats = [list(f) for f in feats]
    message = {'model_was_cashed': was_cashed, 'runtime': {'total': time.time() - t1, 'core': time.time() - t2, 'per_point': (time.time() - t2) / len(payload['rowcols'])}}

    # Store
    k = Key(bucket)
    k.key = payload['outputkey']
    k.set_contents_from_string(json.dumps(feats))

    return message

def train_robot(payload):
    print "Training robot."

    

def classify_image(payload):
    print "Classifying image."




def _download_models(name):
    conn = boto.connect_s3()
    bucket = conn.get_bucket('spacer-tools')
    for suffix in ['.deploy.prototxt', '.caffemodel']:
        was_cashed = _download_file(bucket, name + suffix, '../models/' + name + suffix)
    return was_cashed


def _download_file(bucket, keystring, destination):
    
    if not os.path.isfile(destination):
        print "downloading {}".format(keystring)
        key = Key(bucket, keystring)
        key.get_contents_to_filename(destination)
        return False
    else:
        return True
    
