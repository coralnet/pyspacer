import json
import os
import pickle

import time

import boto

import wget


from spacer import config
from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    ImageLabels, \
    TrainClassifierReturnMsg

from spacer.extract_features import feature_extractor_factory
from spacer.train_classifier import trainer_factory

from spacer.storage import storage_factory


def extract_features_task(msg: ExtractFeaturesMsg) -> ExtractFeaturesReturnMsg:

    print("-> Extracting features for image pk:{}.".format(msg.pk))

    storage = storage_factory(msg.storage_type, msg.bucketname)
    extractor = feature_extractor_factory(msg, storage)
    features, return_message = extractor()
    storage.store_string(msg.outputkey, json.dumps(features.serialize()))
    return return_message


def train_classifier_task(msg: TrainClassifierMsg) -> TrainClassifierReturnMsg:
    print("Training classifier pk:{}.".format(msg.pk))

    storage = storage_factory(msg.storage_type, msg.bucketname)

    trainer = trainer_factory(msg, storage)

    # Do the actual training
    clf, val_results, return_message = trainer()

    # Store
    storage.store_classifier(msg.model_key, clf)
    storage.store_string(msg.valresult_key,
                         json.dumps(val_results.serialize()))
    return return_message


def deploy(payload):

    try:
        t1 = time.time()

        # Make sure the right model and prototxt are available locally.
        was_cashed = _download_nets(payload['modelname'])

        local_impath = os.path.basename(payload['im_url'])

        wget.download(payload['im_url'], local_impath)

        # Setup caffe
        caffe.set_mode_cpu()
        net = caffe.Net(
            str(os.path.join(config.LOCAL_MODEL_PATH, payload['modelname'] + '.deploy.prototxt')),
            str(os.path.join(config.LOCAL_MODEL_PATH, payload['modelname'] + '.caffemodel')),
            caffe.TEST)

        # Set parameters
        pyparams = {'im_mean': [128, 128, 128],
                    'scaling_method': 'scale',
                    'scaling_factor': 1,
                    'crop_size': 224,
                    'batch_size': 10}

        imlist = [local_impath]
        imdict = {
            local_impath: ([], 100)
        }
        for row, col in payload['rowcols']:
            imdict[local_impath][0].append((row, col, 1))

        # Run
        t2 = time.time()
        (_, _, feats) = classify_from_patchlist(imlist, imdict, pyparams, net, scorelayer='fc7')

        # Download the image to be processed.
        conn = boto.connect_s3()
        bucket = conn.get_bucket(payload['bucketname'], validate=True)
        key = bucket.get_key(payload['model'])

        model = pickle.loads(key.get_contents_as_string(), fix_imports=True, encoding='latin1')

        scores = model.predict_proba(feats)

        message = {
            'model_was_cashed': was_cashed,
            'runtime': {
                'total': time.time() - t1,
                'core': time.time() - t2,
                'per_point': (time.time() - t2) / len(payload['rowcols'])
            },
            'scores': [list(score) for score in scores],
            'classes': list(model.classes_),
            'ok': 1
        }
    except Exception as e:

        # For deploy calls we don't use the error queue, but instead return the error message to the standard
        # return queue.
        message = {
            'ok': 0,
            'error': repr(e)
        }

    return message
