"""
Utility methods for training classifiers.
"""

from __future__ import annotations
import random
import string
from collections.abc import Generator
from logging import getLogger
from typing import List, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from spacer import config
from spacer.data_classes import ImageLabels, ImageFeatures
from spacer.exceptions import RowColumnMismatchError
from spacer.messages import DataLocation

logger = getLogger(__name__)


# Implicit type alias; revisit in Python 3.10
# https://peps.python.org/pep-0613/
FeatureLabelPair = Tuple[np.ndarray, int]
FeatureLabelBatch = Tuple[List[np.ndarray], List[int]]


def train(train_labels: ImageLabels,
          ref_labels: ImageLabels,
          feature_loc: DataLocation,
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
        refx, refy = load_batch_data(ref_labels, feature_loc)

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
            for x, y in load_data_as_mini_batches(
                labels=train_labels, feature_loc=feature_loc,
                random_state=epoch,
            ):
                clf.partial_fit(x, y, classes=classes_list)

            ref_acc.append(calc_acc(refy, clf.predict(refx)))
            logger.debug(f"Epoch {epoch}, acc: {ref_acc[-1]}")

    with config.log_entry_and_exit("calibration"):
        clf_calibrated = CalibratedClassifierCV(clf, cv="prefit")
        clf_calibrated.fit(refx, refy)

    return clf_calibrated, ref_acc


def evaluate_classifier(clf: CalibratedClassifierCV,
                        labels: ImageLabels,
                        feature_loc: DataLocation) -> tuple[list, list, list]:
    """ Evaluates classifier on data """
    scores, gts, ests = [], [], []

    for image_key in labels.image_keys:

        feature_loc.key = image_key
        image_labels_data = labels.data[image_key]

        pairs = list(
            load_image_data(image_labels_data, feature_loc))

        # List of pairs -> pair of lists
        x, y = zip(*pairs)

        scores.extend(clf.predict_proba(x).max(axis=1).tolist())
        ests.extend(clf.predict(x))
        gts.extend(y)

    assert len(gts) > 0, (
        "The validation set should have been checked for emptiness during"
        " label preprocessing.")

    return gts, ests, scores


def load_image_data(labels_data: list[tuple[int, int, int]],
                    feature_loc: DataLocation) \
        -> Generator[FeatureLabelPair, None, None]:
    """
    Loads a feature vector and labels of a single image, and generates
    element-matching pairs.
    """
    # Load features.
    features = ImageFeatures.load(feature_loc)

    return match_features_and_labels(features, labels_data, feature_loc.key)


def load_batch_data(labels: ImageLabels,
                    feature_loc: DataLocation) \
        -> FeatureLabelBatch:
    """
    Loads features and labels, and builds element-matching lists
    for use with methods such as CalibratedClassifierCV.fit().
    """
    batch = []

    for image_key in labels.image_keys:

        feature_loc.key = image_key
        image_labels_data = labels.data[image_key]

        batch.extend(
            load_image_data(image_labels_data, feature_loc))

    assert len(batch) > 0, (
        "We only ever expect labels to be a non-empty ref set.")

    # List of pairs -> pair of lists
    x, y = zip(*batch)
    return x, y


def load_data_as_mini_batches(labels: ImageLabels,
                              feature_loc: DataLocation,
                              # For seeding the randomizer to get repeatable
                              # results.
                              random_state: int) \
        -> Generator[FeatureLabelBatch, None, None]:
    """
    Loads features and labels, and generates batches of
    element-matching pairs.

    Since this is a generator, it does not have to load all feature
    vectors in memory at the same time; only as many as will fit in
    a batch.
    Note that a single image's features may straddle multiple batches.
    """
    image_keys = labels.image_keys

    # Shuffle the order of images.
    np.random.seed(random_state)
    np.random.shuffle(image_keys)

    current_batch = []

    for image_key in image_keys:

        feature_loc.key = image_key
        image_labels_data = labels.data[image_key]

        for point_feature, label in load_image_data(
                image_labels_data, feature_loc):

            current_batch.append((point_feature, label))

            if len(current_batch) >= config.TRAINING_BATCH_LABEL_COUNT:
                # List of pairs -> pair of lists
                x, y = zip(*current_batch)
                yield x, y
                current_batch = []

    if len(current_batch) > 0:
        # Last batch
        x, y = zip(*current_batch)
        yield x, y


def match_features_and_labels(features: ImageFeatures,
                              labels_data: list[tuple[int, int, int]],
                              image_key: str) \
        -> Generator[FeatureLabelPair, None, None]:

    # Sanity check
    if features.valid_rowcol:
        # With new data structure just check that the sets of row, col
        # given by the labels are available in the features.
        rc_features_set = set([(pf.row, pf.col) for pf in
                               features.point_features])
        rc_labels_set = set([(row, col) for (row, col, _) in labels_data])

        if not rc_labels_set.issubset(rc_features_set):
            difference_set = rc_labels_set.difference(rc_features_set)
            example_rc = next(iter(difference_set))
            raise RowColumnMismatchError(
                f"{image_key}: The labels' row-column positions don't match"
                f" those of the feature vector (example: {example_rc}).")
    else:
        # With legacy data structure check that length is the same.
        label_count = len(labels_data)
        feature_count = len(features.point_features)

        if not label_count == feature_count:
            raise RowColumnMismatchError(
                f"{image_key}: The number of labels ({label_count}) doesn't"
                f" match the number of extracted features ({feature_count}).")

    if features.valid_rowcol:
        for row, col, label in labels_data:
            yield features[(row, col)], label
    else:
        # For legacy features, we didn't store the row, col information.
        # Instead rely on ordering.
        for (_, _, label), point_feature in zip(labels_data,
                                                features.point_features):
            yield point_feature.data, label


def calc_acc(gt: list[int], est: list[int]) -> float:
    """
    Calculate the accuracy of (agreement between) two integer valued list.
    """
    if len(gt) == 0 or len(est) == 0:
        raise TypeError('Inputs can not be empty')

    if not len(gt) == len(est):
        raise ValueError('Input gt and est must have the same length')

    for g in gt:
        if not int(g) == g:
            raise TypeError('Input gt must be an array of ints')

    for e in est:
        if not int(e) == e:
            raise TypeError('Input est must be an array of ints')

    return float(sum([(g == e) for (g, e) in zip(gt, est)])) / len(gt)


def make_random_data(im_count: int,
                     class_list: list[int],
                     points_per_image: int,
                     feature_dim: int,
                     feature_loc: DataLocation) -> ImageLabels:
    """
    Utility method for testing that generates an ImageLabels instance
    complete with stored ImageFeatures.
    """
    data = {}
    for _ in range(im_count):

        # Generate random features (using labels to draw from a Gaussian).
        point_labels = np.random.choice(class_list, points_per_image).tolist()

        # Make sure all classes are present
        point_labels[:len(class_list)] = class_list
        feats = ImageFeatures.make_random(point_labels, feature_dim)

        # Generate a random string as imkey.
        imkey = ''.join(random.choice(string.ascii_uppercase + string.digits)
                        for _ in range(20))

        # Store
        feature_loc.key = imkey
        feats.store(feature_loc)
        data[imkey] = [
            (pf.row, pf.col, pl) for pf, pl in
            zip(feats.point_features, point_labels)
        ]
    return ImageLabels(data)
