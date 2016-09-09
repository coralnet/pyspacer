import sys
sys.path.append('/root/caffe')

import caffe
import os
import boto
import json

import beijbom_vision_lib.caffe.tools as bct
import coral_lib.patch.tools as cpt

from boto.s3.key import Key

def extract_features(payload):
    print "Extracting features for {}.".format(payload['imkey'])
    
    # Make sure the right model and prototxt are available locally.
    _download_models(payload['modelname'])

    # Download the image to be processed.
    conn = boto.connect_s3()
    bucket = conn.get_bucket(payload['bucketname'], validate=True)

    key = bucket.get_key(payload['imkey'])
    basename = os.path.basename(payload['imkey'])
    key.get_contents_to_filename(basename)

    # Setup caffe
    caffe.set_mode_cpu()
    net = caffe.Net(payload['modelname'] + '.deploy.prototxt', payload['modelname'] + '.caffemodel', caffe.TEST)

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
    (_, _, feats) = cpt.classify_from_patchlist(imlist, imdict, pyparams, net, scorelayer = 'fc7')
    feats = [list(f) for f in feats]

    # Store
    k = Key(bucket)
    k.key = payload['outputkey']
    k.set_contents_from_string(json.dumps(feats))

    return 1, {'runtime', 3}

def train_robot(payload):
    print "Training robot."

    

def classify_image(payload):
    print "Classifying image."








def _download_models(name):
    conn = boto.connect_s3()
    bucket = conn.get_bucket('spacer-tools')
    for suffix in ['.deploy.prototxt', '.caffemodel']:
        _download_file(bucket, name + suffix, name + suffix)



def _download_file(bucket, keystring, destination):
    
    if not os.path.isfile(destination):
        print "downloading {}".format(keystring)
        key = Key(bucket, keystring)
        key.get_contents_to_filename(destination)
    
