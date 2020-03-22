"""
Defines the highest level methods for completing tasks.
"""

import json
import time

import numpy as np

from spacer.data_classes import ImageLabels, ImageFeatures, SpacerImage
from spacer.extract_features import feature_extractor_factory
from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    ClassifyFeaturesMsg, \
    ClassifyImageMsg, \
    ClassifyReturnMsg
from spacer.storage import store, load
from spacer.train_classifier import trainer_factory


def extract_features(msg: ExtractFeaturesMsg) -> ExtractFeaturesReturnMsg:

    print("-> Extracting features for job:{}.".format(msg.job_token))
    extractor = feature_extractor_factory(msg.feature_extractor_name)
    img = load(msg.image_loc, 'image')
    features, return_msg = extractor(img, msg.rowcols)
    store(msg.feature_loc, json.dumps(features.serialize()), 'str')
    return return_msg


def train_classifier(msg: TrainClassifierMsg) -> TrainClassifierReturnMsg:

    print("Training classifier pk:{}.".format(msg.job_token))
    trainer = trainer_factory(msg.trainer_name)

    # Do the actual training
    clf, val_results, return_message = trainer(
        ImageLabels.load(msg.traindata_loc),
        ImageLabels.load(msg.traindata_loc),
        msg.nbr_epochs,
        [load(loc, 'clf') for loc in msg.previous_model_locs],
        msg.features_loc
    )

    # Store
    store(msg.model_loc, clf, 'clf')
    store(msg.valresult_loc, json.dumps(val_results.serialize()), 'str')

    return return_message


def classify_features(msg: ClassifyFeaturesMsg) -> ClassifyReturnMsg:

    t0 = time.time()
    features = ImageFeatures.load(msg.feature_loc)

    clf = load(msg.classifier_loc, 'clf')

    scores = [(pf.row, pf.col, clf.predict_proba(np.array(pf.data))) for
              pf in features.point_features]

    # Return
    return ClassifyReturnMsg(
        runtime=time.time() - t0,
        scores=scores,
        classes=list(clf.classes_))


def classify_image(msg: ClassifyImageMsg) -> ClassifyReturnMsg:

    t0 = time.time()

    # Download image
    img = load(msg.image_loc, 'image')

    # Extract features
    extractor = feature_extractor_factory(msg.feature_extractor_name)
    features, _ = extractor(img, msg.rowcols)

    # Classify
    clf = load(msg.classifier_loc, 'clf')
    scores = [(row, col, clf.predict_proba(features.get_array((row, col))).
               tolist()) for row, col in msg.rowcols]

    # Return
    return ClassifyReturnMsg(
        runtime=time.time() - t0,
        scores=scores,
        classes=list(clf.classes_))
