"""
Defines the highest level methods for completing tasks.
"""

import json
import os
import time
from PIL import Image

import wget

from spacer.extract_features import feature_extractor_factory
from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    ClassifyImageMsg, \
    ClassifyReturnMsg
    #ClassifyFeatMsg, \
    #ClassifyReturnMsg, \

from spacer.storage import store, load, storage_factory
from spacer.train_classifier import trainer_factory


def extract_features(msg: ExtractFeaturesMsg) -> ExtractFeaturesReturnMsg:

    print("-> Extracting features for job:{}.".format(msg.job_token))
    storage = storage_factory(msg.storage_type, msg.bucketname)
    extractor = feature_extractor_factory(msg.feature_extractor_name)
    features, return_msg = extractor(storage.load_image(msg.imkey),
                                     msg.rowcols)
    storage.store_string(msg.outputkey, json.dumps(features.serialize()))
    return return_msg


def train_classifier(msg: TrainClassifierMsg) -> TrainClassifierReturnMsg:

    print("Training classifier pk:{}.".format(msg.pk))
    storage = storage_factory(msg.storage_type, msg.bucketname)
    trainer = trainer_factory(msg.trainer_name)

    # Do the actual training
    clf, val_results, return_message = trainer(
        msg.traindata_key,
        msg.valdata_key,
        msg.nbr_epochs,
        msg.pc_models_key,
        storage
    )

    # Store
    storage.store_classifier(msg.model_key, clf)
    storage.store_string(msg.valresult_key,
                         json.dumps(val_results.serialize()))
    return return_message

#def classify_features(msg: ClassifyFeatMsg) -> ClassifyReturnMsg:
#    pass


def deploy(msg: ClassifyImageMsg) -> ClassifyReturnMsg:
    """ Deploy is a combination of feature extractor and classification. """

    t0 = time.time()

    # Download image
    img = load(msg.image_loc, 'image')

    # Extract features
    extractor = feature_extractor_factory(msg.feature_extractor_name)
    features, feats_return_message = extractor(img, msg.rowcols)

    # Classify
    clf = load(msg.classifier_loc, 'clf')
    scores = [(row, col, clf.predict_proba(features.get_array((row, col))).
               tolist()) for row, col in msg.rowcols]

    # Return
    return ClassifyReturnMsg(
        model_was_cached=feats_return_message.model_was_cashed,
        runtime=time.time() - t0,
        scores=scores,
        classes=list(clf.classes_))
