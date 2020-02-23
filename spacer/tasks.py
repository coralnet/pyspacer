import json
import os
import time

import wget

from spacer.extract_features import feature_extractor_factory
from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    DeployMsg, \
    DeployReturnMsg
from spacer.storage import storage_factory
from spacer.train_classifier import trainer_factory


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


def deploy(msg: DeployMsg) -> DeployReturnMsg:
    """ Deploy is a combination of feature extractor and classification. """

    t0 = time.time()

    # Download image
    local_impath = os.path.basename(msg.im_url)
    wget.download(msg.im_url, local_impath)

    # Extract features
    extract_features_msg = ExtractFeaturesMsg(
        pk=0,
        modelname=msg.feature_extractor_name,
        bucketname='',
        imkey=local_impath,
        rowcols=msg.rowcols,
        outputkey='',
        storage_type='filesystem'
    )

    storage = storage_factory(extract_features_msg.storage_type, '')
    extractor = feature_extractor_factory(extract_features_msg, storage)
    features, feats_return_message = extractor()

    # Classify
    storage = storage_factory('s3', msg.bucketname)
    clf = storage.load_classifier(msg.classifier_key)

    scores = [clf.predict_proba(features[(row, col)])
              for row, col in msg.rowcols]

    # Return
    return DeployReturnMsg(
        model_was_cached=feats_return_message.model_was_cashed,
        runtime=time.time() - t0,
        scores=[score.tolist() for score in scores],
        classes=list(clf.classes_))
