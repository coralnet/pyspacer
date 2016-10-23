import sys
sys.path.append('/root/caffe/')

import caffe
import os
import boto
import json
import time
import random
import pickle
import random

import numpy as np

import beijbom_vision_lib.caffe.tools as bct
import beijbom_vision_lib.misc.tools as bmt
import coral_lib.patch.tools as cpt

from boto.s3.key import Key

from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

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
    
    # SETUP
    #
    conn = boto.connect_s3()
    bucket = conn.get_bucket(payload['bucketname'], validate=True)
    k = Key(bucket)
    k.key = payload['traindata']
    traindict = json.loads(k.get_contents_as_string())

    ## TRAIN A MODEL
    #
    starttime = time.time()
    clf, refacc = _do_training(traindict, int(payload['nbr_epochs']), bucket)
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
    k.set_contents_from_string(json.dumps({'scores':scores, 'gt':gt, 'est':est, 'classes':list(clf.classes_)}))

    ## FINALLY, EVALUATE ALL PREVIOUS MODELS ON THE VAL SET TO DETERMINE WHETER TO KEEP THE NEW MODEL
    #
    ps_accs = []
    for pc_model in payload['pc_models']:
        k.key = pc_model
        this_clf = pickle.loads(k.get_contents_as_string())
        gt, est, scores = _evaluate_classifier(this_clf, valdict.keys(), valdict, bucket)
        ps_accs.append(bmt.acc(gt, est))

    # Return
    return {'runtime': runtime, 'refacc': refacc, 'acc': valacc, 'pc_accs': ps_accs}


def _do_training(traindict, nbr_epochs, bucket):
  
    def get_unique_classes(keylist):
        labels = set()
        for imkey in keylist:
            labels = labels.union(set(traindict[imkey]))
        return labels
    

    # Calculate max nbr images to keep in memory (based on 20000 samples total).
    samples_per_image = len(traindict[traindict.keys()[0]])
    max_imgs_in_memory = 20000 / samples_per_image

    # Make train and ref split. Reference set is here a hold-out part of the train-data portion.
    # Purpose of refset is to 1) know accuracy per epoch and 2) calibrate classifier output scores.
    # We call it 'ref' to disambiguate from the actual validation set of the source.
    imkeys = traindict.keys()
    refset = imkeys[::10]
    random.shuffle(refset)
    refset = refset[:max_imgs_in_memory] # make sure we don't go over the memory limit.
    trainset = list(set(imkeys) - set(refset))
    print "trainset: {}, valset: {} images".format(len(trainset), len(refset))

    # Figure out # images per mini-batch and batches per epoch.
    images_per_minibatch = min(max_imgs_in_memory, len(trainset))
    n = int(np.ceil(len(trainset) / float(images_per_minibatch)))
    print "Using {} images per mini-batch and {} mini-batches per epoch".format(images_per_minibatch, n)

    # Identify classes common to both train and val. This will be our labelset for the training.
    trainclasses = get_unique_classes(trainset) 
    refclasses = get_unique_classes(refset)
    classes = list(trainclasses.intersection(refclasses))
    print "trainset: {}, valset: {}, common: {} labels".format(len(trainclasses), len(refclasses), len(classes))
    
    # Load reference data (must hold in memory for the calibration)
    print "Loading reference data."
    refx, refy = _load_mini_batch(traindict, refset, classes, bucket)

    # Initialize classifier and ref set accuracy list
    print "Online training..."
    clf = SGDClassifier(loss = 'log', average = True)
    refacc = []
    for epoch in range(nbr_epochs):
        print "Epoch {}".format(epoch)
        random.shuffle(trainset)
        mini_batches = _chunkify(trainset, n)
        for mb in mini_batches:
            x, y = _load_mini_batch(traindict, mb, classes, bucket)
            clf.partial_fit(x, y, classes = classes)
        refacc.append(bmt.acc(refy, clf.predict(refx)))
        print "acc: {}".format(refacc[-1])
    
    print "Calibrating."
    clf_calibrated = CalibratedClassifierCV(clf, cv = "prefit")
    clf_calibrated.fit(x, y)

    return clf_calibrated, refacc        

def _evaluate_classifier(clf, imkeys, gtdict, bucket):
    """
    Return the accuracy of classifier "clf" evaluated on "imkeys"
    with ground truth given in "gtdict". Features are fetched from S3 "bucket".
    """
    scores = []
    gt = []
    classes = list(clf.classes_)
    for imkey in imkeys:
        x, y = _load_data(gtdict, imkey, classes, bucket)
        scores.extend(clf.predict_proba(x))
        # Convert the ground truth to index not actual class id.
        y_index = [classes.index(ymember) for ymember in y]
        gt.extend(y_index)
    scores = [list(score) for score in scores]
    # Est also given as index not actual class id. 
    est = [np.argmax(score) for score in scores]
    return gt, est, scores

def _chunkify(lst, n):
    return [ lst[i::n] for i in xrange(n) ]

def _load_data(labeldict, imkey, classes, bucket):
    """
    load feats from S3 and returns them along with the ground truth labels
    """
    key = bucket.get_key(imkey)

    # Load featutes
    x = json.loads(key.get_contents_as_string())

    # Load labels
    y = labeldict[imkey]

    # Remove samples for which the label is not in classes.
    x, y = zip(*[(xmember, ymember) for xmember, ymember in zip(x, y) if ymember in classes]) 
    return x, y
    
def _load_mini_batch(labeldict, imkeylist, classes, bucket):
    """
    An interator over _load_data.
    """
    x, y = [], []
    for imkey in imkeylist:
        thisx, thisy = _load_data(labeldict, imkey, classes, bucket)
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
    
