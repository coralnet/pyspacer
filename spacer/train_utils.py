"""
Utility methods for training classifiers.
"""

from __future__ import annotations
import random
import string
from logging import getLogger

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from spacer import config
from spacer.data_classes import (
    DataLocation, ImageLabels, ImageFeatures, LabelId)

logger = getLogger(__name__)


def train(train_labels: ImageLabels,
          ref_labels: ImageLabels,
          nbr_epochs: int,
          clf_type: str) -> tuple[CalibratedClassifierCV, list[float]]:

    logger.debug(
        f"Data sets:"
        f" Train = {len(train_labels)} images,"
        f" {train_labels.label_count} labels;"
        f" Ref = {len(ref_labels)} images,"
        f" {ref_labels.label_count} labels")
    logger.debug(
        f"Mini-batch size: {config.TRAINING_BATCH_LABEL_COUNT} labels")

    # train_labels and ref_labels should already be trimmed down to the
    # classes common to both, so the classes_set from either one should be the
    # set used for training.
    classes_list = list(ref_labels.classes_set)

    # Load reference data (must hold in memory for the calibration)
    with config.log_entry_and_exit("loading of reference data"):
        refx, refy = ref_labels.load_all_data()

    # Initialize classifier and ref set accuracy list
    with config.log_entry_and_exit("training using " + clf_type):
        if clf_type == 'MLP':
            if train_labels.label_count >= 50000:
                hls, lr = (200, 100), 1e-4
            else:
                hls, lr = (100,), 1e-3
            clf = MLPClassifier(hidden_layer_sizes=hls, learning_rate_init=lr)
        else:
            clf = SGDClassifier(loss='log_loss', average=True, random_state=0)

        ref_acc = []

        for epoch in range(nbr_epochs):
            for x, y in train_labels.load_data_in_batches(
                batch_size=config.TRAINING_BATCH_LABEL_COUNT,
                random_seed=epoch,
            ):
                clf.partial_fit(x, y, classes=classes_list)

            ref_acc.append(calc_acc(refy, clf.predict(refx)))
            logger.debug(f"Epoch {epoch}, acc: {ref_acc[-1]}")

    with config.log_entry_and_exit("calibration"):
        clf_calibrated = CalibratedClassifierCV(clf, cv="prefit")
        clf_calibrated.fit(refx, refy)

    return clf_calibrated, ref_acc


def evaluate_classifier(clf: CalibratedClassifierCV,
                        labels: ImageLabels) -> tuple[list, list, list]:
    """ Evaluates classifier on data """
    scores, gts, ests = [], [], []

    # In each iteration, we get:
    # List of point features, list of corresponding ground-truth labels.
    for batch_x, batch_y in labels.load_data_in_batches():
        scores.extend(clf.predict_proba(batch_x).max(axis=1).tolist())
        ests.extend(clf.predict(batch_x))
        gts.extend(batch_y)

    assert len(gts) > 0, (
        "The validation set should have been checked for emptiness during"
        " label preprocessing.")

    return gts, ests, scores


def calc_acc(gt: list, est: list) -> float:
    """
    Calculate the accuracy of (agreement between) two lists whose elements can
    be tested for equality.
    """
    if len(gt) == 0 or len(est) == 0:
        raise ValueError('Inputs can not be empty')

    if not len(gt) == len(est):
        raise ValueError('Input gt and est must have the same length')

    return float(sum([(g == e) for (g, e) in zip(gt, est)])) / len(gt)


def make_random_data(im_count: int,
                     class_list: list[LabelId],
                     points_per_image: int,
                     feature_dim: int,
                     feature_loc_base: DataLocation,
                     feature_loc_keys: list[str] | None = None) -> ImageLabels:
    """
    Utility method for testing that generates an ImageLabels instance
    complete with stored ImageFeatures.

    Feature DataLocations are constructed using storage_type and bucket_name
    of feature_loc_base, and keys of feature_loc_keys. This should be a more
    convenient format for defining test data, compared to a DataLocation list.
    """
    all_labels = ImageLabels()
    for i in range(im_count):

        # Generate random features (using labels to draw from a Gaussian).
        point_labels = np.random.choice(class_list, points_per_image).tolist()

        # Make sure all classes are present
        point_labels[:len(class_list)] = class_list
        feats = ImageFeatures.make_random(point_labels, feature_dim)

        if feature_loc_keys is None:
            # Generate a random string as feature_loc_key.
            feature_loc_key = ''.join(
                random.choice(string.ascii_uppercase + string.digits)
                for _ in range(20))
        else:
            # Use the keys passed into this function.
            # feature_loc_keys must contain at least im_count elements.
            feature_loc_key = feature_loc_keys[i]

        # Store
        feature_loc = DataLocation(
            storage_type=feature_loc_base.storage_type,
            bucket_name=feature_loc_base.bucket_name,
            key=feature_loc_key,
        )
        feats.store(feature_loc)
        this_image_labels = [
            (pf.row, pf.col, pl) for pf, pl in
            zip(feats.point_features, point_labels)
        ]
        all_labels.add_image(feature_loc, this_image_labels)
    return all_labels
