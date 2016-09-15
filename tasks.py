import sys
sys.path.append('/root/caffe/')

import caffe
import os
import boto
import json
import time
import random
import pickle

import numpy as np

import beijbom_vision_lib.caffe.tools as bct
import beijbom_vision_lib.misc.tools as bmt
import coral_lib.patch.tools as cpt

from boto.s3.key import Key

from sklearn import linear_model

def extract_features(payload):
    print "Extracting features for image pk:{}.".format(payload['pk'])
    t1 = time.time()

    # Make sure the right model and prototxt are available locally.
    was_cashed = _download_nets(payload['modelname'])

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

def train_classifier(payload):
    print "Training classifier pk:{}.".format(payload['pk'])
    print payload
    # Load
    conn = boto.connect_s3()
    bucket = conn.get_bucket(payload['bucketname'], validate=True)
    k = Key(bucket)
    k.key = payload['traindata']
    traindict = json.loads(k.get_contents_as_string())

    ## TRAIN A MODEL
    #
    starttime = time.time()
    clf, refacc = _do_training(traindict, payload['classes'], bucket)
    runtime = time.time() - starttime

    # Store
    k.key = payload['model']
    k.set_contents_from_string(pickle.dumps(clf))

    ## EVALUATE ON THE VALIDATION SET
    #
    key = bucket.get_key(payload['valdata'])
    valdict = json.loads(key.get_contents_as_string())
    gt, est, scores = _evaluate_classifier(clf, valdict.keys(), valdict, bucket)
    valacc = bmt.acc(gt, est)

    # Store
    k.key = payload['valresult']
    k.set_contents_from_string(json.dumps({'gt':gt, 'est':est, 'scores':scores}))

    ## FINALLY, EVALUATE ALL PREVIOUS MODELS ON THE VAL SET TO DETERMINE WHETER TO KEEP THE NEW MODEL
    ps_accs = []
    for pc_model in payload['pc_models']:
        k.key = pc_model
        this_clf = pickle.loads(k.get_contents_as_string())
        gt, est, scores = _evaluate_classifier(this_clf, valdict.keys(), valdict, bucket)
        ps_accs.append(bmt.acc(gt, est))

    # Return
    return {'runtime': runtime, 'refacc': refacc, 'acc': valacc, 'pc_accs': ps_accs}


def classify_image(payload):
    print "Classifying image."


def _do_training(traindict, classes, bucket):

    # Figure out # images per mini-batch.
    samples_per_image = len(traindict[traindict.keys()[0]])
    images_per_minibatch = 10000 / samples_per_image

    # Make train and ref split. Reference set is here a hold-out part of the train-data portion. Called 'ref' to disambiguate from the actual validation set of the source.
    imkeys = traindict.keys()
    refset = imkeys[::10]
    trainset = list(set(imkeys) - set(refset))

    # Number of mini-batches per epoch.
    n = len(trainset) / images_per_minibatch + 1

    # Initialize classifier and ref set accuracy list
    clf = linear_model.SGDClassifier(loss = 'log', average = True)
    refacc = []

    print "Trainstats:", images_per_minibatch, n
   
    for epoch in range(10):
        print "Epoch {}".format(epoch),
        random.shuffle(trainset)
        mini_batches = _chunkify(trainset, n)
        for mb in mini_batches:
            x, y = _load_mini_batch(traindict, mb, bucket)
            clf.partial_fit(x, y, classes = classes)
        gt, est, _ = _evaluate_classifier(clf, refset, traindict, bucket)
        refacc.append(bmt.acc(gt, est))
        print "acc: {}".format(refacc[-1])
    return clf, refacc 


def _evaluate_classifier(clf, imkeys, gtdict, bucket):
    """
    Return the accuracy of classifier "clf" evaluated on "imkeys"
    with ground truth given in "gtdict". Features are fetched from S3 "bucket".
    """
    scores = []
    gt = []
    for imkey in imkeys:
        x, y = _load_data(gtdict, imkey, bucket)
        scores.extend(clf.predict_proba(x))
        gt.extend(y)
    scores = [list(score) for score in scores]
    est = [np.argmax(score) for score in scores]
    return gt, est, scores

def _chunkify(lst, n):
    return [ lst[i::n] for i in xrange(n) ]

def _load_data(gtdict, imkey, bucket):
    """
    load feats from S3 and returns them along with the ground truth labels
    """
    key = bucket.get_key(imkey)
    x = json.loads(key.get_contents_as_string())
    y = gtdict[imkey]
    return x, y
    
def _load_mini_batch(traindict, imkeylist, bucket):

    x, y = [], []
    for imkey in imkeylist:
        thisx, thisy = _load_data(traindict, imkey, bucket)
        x.extend(thisx)
        y.extend(thisy)
    return x, y

def _download_nets(name):
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
    
