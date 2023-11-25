"""
Utility methods for training classifiers.
"""

import logging
import random
import string
from typing import Tuple, List

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from spacer import config
from spacer.data_classes import ImageLabels, ImageFeatures
from spacer.exceptions import RowColumnMismatchError
from spacer.messages import DataLocation


def train(labels: ImageLabels,
          feature_loc: DataLocation,
          nbr_epochs: int,
          clf_type: str) -> Tuple[CalibratedClassifierCV, List[float]]:

    if len(labels) < config.MIN_TRAINIMAGES:
        raise ValueError('Not enough training samples.')

    # Calculate max nbr images to keep in memory (for 5000 samples total).
    max_imgs_in_memory = 5000 // labels.samples_per_image

    # Make train and ref split.
    # Reference set is here a hold-out part of the train-data portion.
    # Purpose of reference set is to
    # 1) know accuracy per epoch
    # 2) calibrate classifier output scores.
    # We call it "reference" to disambiguate from the validation set.
    ref_set = labels.image_keys[::10]
    np.random.shuffle(ref_set)
    ref_set = ref_set[:max_imgs_in_memory]  # Enforce memory limit.
    train_set = list(set(labels.image_keys) - set(ref_set))
    logging.info("Trainset: {}, valset: {} images".
                 format(len(train_set), len(ref_set)))

    # Figure out # images per batch and batches per epoch.
    images_per_batch, batches_per_epoch = \
        calc_batch_size(max_imgs_in_memory, len(train_set))
    logging.info("Using {} images per mini-batch and {} mini-batches per "
                 "epoch".format(images_per_batch, batches_per_epoch))

    # Identify classes common to both train and val.
    # This will be our labelset for the training.
    trainclasses = labels.unique_classes(train_set)
    refclasses = labels.unique_classes(ref_set)
    classes = list(trainclasses.intersection(refclasses))
    logging.info("Trainset: {}, valset: {}, common: {} labels".format(
        len(trainclasses), len(refclasses), len(classes)))
    if len(classes) == 1:
        raise ValueError('Not enough classes to do training (only 1)')

    # Load reference data (must hold in memory for the calibration)
    with config.log_entry_and_exit("loading of reference data"):
        refx, refy = load_batch_data(labels, ref_set, classes, feature_loc)

    # Initialize classifier and ref set accuracy list
    with config.log_entry_and_exit("training using " + clf_type):
        if clf_type == 'MLP':
            if len(train_set) * labels.samples_per_image >= 50000:
                hls, lr = (200, 100), 1e-4
            else:
                hls, lr = (100,), 1e-3
            clf = MLPClassifier(hidden_layer_sizes=hls, learning_rate_init=lr)
        else:
            clf = SGDClassifier(loss='log_loss', average=True, random_state=0)
        ref_acc = []
        for epoch in range(nbr_epochs):
            np.random.shuffle(train_set)
            mini_batches = chunkify(train_set, batches_per_epoch)
            for mb in mini_batches:
                x, y = load_batch_data(labels, mb, classes, feature_loc)
                clf.partial_fit(x, y, classes=classes)
            ref_acc.append(calc_acc(refy, clf.predict(refx)))
            logging.info("Epoch {}, acc: {}".format(epoch, ref_acc[-1]))

    with config.log_entry_and_exit("calibration"):
        clf_calibrated = CalibratedClassifierCV(clf, cv="prefit")
        clf_calibrated.fit(refx, refy)

    return clf_calibrated, ref_acc


def evaluate_classifier(clf: CalibratedClassifierCV,
                        labels: ImageLabels,
                        classes: List[int],
                        feature_loc: DataLocation) -> Tuple[List, List, List]:
    """ Evaluates classifier on data """
    scores, gts, ests = [], [], []
    for imkey in labels.image_keys:
        x, y = load_image_data(labels, imkey, classes, feature_loc)
        if len(x) > 0:
            scores.extend(clf.predict_proba(x).max(axis=1).tolist())
            ests.extend(clf.predict(x))
            gts.extend(y)

    if len(gts) == 0:
        raise ValueError('Not enough data in validation set.')

    return gts, ests, scores


def chunkify(lst: List,
             nbr_chunks: int) -> List:
    return [lst[i::nbr_chunks] for i in range(nbr_chunks)]


def calc_batch_size(max_imgs_in_memory: int,
                    train_set_size: int) -> Tuple[int, int]:
    images_per_batch = min(max_imgs_in_memory, train_set_size)
    batches_per_epoch = int(np.ceil(train_set_size / images_per_batch))
    return images_per_batch, batches_per_epoch


def load_image_data(labels: ImageLabels,
                    imkey: str,
                    classes: List[int],
                    feature_loc: DataLocation) \
        -> Tuple[List[List[float]], List[int]]:
    """
    Loads features and labels for image and matches feature with labels.
    """

    # Load features for this image.
    feature_loc.key = imkey  # Set the relevant key here.
    image_features = ImageFeatures.load(feature_loc)

    # Load row, col, labels for this image.
    image_labels = labels.data[imkey]

    # Sanity check
    if image_features.valid_rowcol:
        # With new data structure just check that the sets of row, col
        # given by the labels are available in the features.
        rc_features_set = set([(pf.row, pf.col) for pf in
                               image_features.point_features])
        rc_labels_set = set([(row, col) for (row, col, _) in image_labels])

        if not rc_labels_set.issubset(rc_features_set):
            difference_set = rc_labels_set.difference(rc_features_set)
            example_rc = next(iter(difference_set))
            raise RowColumnMismatchError(
                f"{imkey}: The labels' row-column positions don't match those"
                f" of the feature vector (example: {example_rc}).")
    else:
        # With legacy data structure check that length is the same.
        label_count = len(image_labels)
        feature_count = len(image_features.point_features)

        if not label_count == feature_count:
            raise RowColumnMismatchError(
                f"{imkey}: The number of labels ({label_count}) doesn't match"
                f" the number of extracted features ({feature_count}).")

    x, y = [], []
    if image_features.valid_rowcol:
        for row, col, label in image_labels:
            if label not in classes:
                # Remove samples for which the label is not in classes.
                continue
            x.append(image_features[(row, col)])
            y.append(label)

    else:
        # For legacy features, we didn't store the row, col information.
        # Instead rely on ordering.
        for (_, _, label), point_feature in zip(image_labels,
                                                image_features.point_features):
            if label not in classes:
                continue
            x.append(point_feature.data)
            y.append(label)

    return x, y


def load_batch_data(labels: ImageLabels,
                    imkeylist: List[str],
                    classes: List[int],
                    feature_loc: DataLocation) \
        -> Tuple[List[List[float]], List[int]]:
    """ Loads features and labels and match them together. """
    x, y = [], []
    for imkey in imkeylist:
        x_, y_ = load_image_data(labels, imkey, classes, feature_loc)
        x.extend(x_)
        y.extend(y_)
    return x, y


def calc_acc(gt: List[int], est: List[int]) -> float:
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
                     class_list: List[int],
                     points_per_image: int,
                     feature_dim: int,
                     feature_loc: DataLocation) -> ImageLabels:
    """
    Utility method for testing that generates an ImageLabels instance
    complete with stored ImageFeatures.
    """
    labels = ImageLabels(data={})
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
        labels.data[imkey] = [
            (pf.row, pf.col, pl) for pf, pl in
            zip(feats.point_features, point_labels)
        ]
    return labels
