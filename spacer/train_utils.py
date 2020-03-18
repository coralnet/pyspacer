"""
Utility methods for training classifiers.
"""

import random
import string
from typing import Tuple, List

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier

from spacer import config
from spacer.data_classes import ImageLabels, ImageFeatures, DataLocation
from spacer.storage import Storage


def train(labels: ImageLabels,
          feature_loc: DataLocation,
          nbr_epochs: int) -> Tuple[CalibratedClassifierCV, List[float]]:

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
    print("-> Trainset: {}, valset: {} images".
          format(len(train_set), len(ref_set)))

    # Figure out # images per batch and batches per epoch.
    images_per_batch, batches_per_epoch = \
        calc_batch_size(max_imgs_in_memory, len(train_set))
    print("-> Using {} images per mini-batch and {} mini-batches per epoch".
          format(images_per_batch, batches_per_epoch))

    # Identify classes common to both train and val.
    # This will be our labelset for the training.
    trainclasses = labels.unique_classes(train_set)
    refclasses = labels.unique_classes(ref_set)
    classes = list(trainclasses.intersection(refclasses))
    print("-> Trainset: {}, valset: {}, common: {} labels".format(
        len(trainclasses), len(refclasses), len(classes)))
    if len(classes) == 1:
        raise ValueError('Not enough classes to do training (only 1)')

    # Load reference data (must hold in memory for the calibration)
    print("-> Loading reference data.")
    refx, refy = load_batch_data(labels, ref_set, classes, feature_loc)

    # Initialize classifier and ref set accuracy list
    print("-> Online training...")
    clf = SGDClassifier(loss='log', average=True, random_state=0)
    ref_acc = []
    for epoch in range(nbr_epochs):
        np.random.shuffle(train_set)
        mini_batches = chunkify(train_set, batches_per_epoch)
        for mb in mini_batches:
            x, y = load_batch_data(labels, mb, classes, storage)
            clf.partial_fit(x, y, classes=classes)
        ref_acc.append(calc_acc(refy, clf.predict(refx)))
        print("-> Epoch {}, acc: {}".format(epoch, ref_acc[-1]))

    print("-> Calibrating.")
    clf_calibrated = CalibratedClassifierCV(clf, cv="prefit")
    clf_calibrated.fit(refx, refy)

    return clf_calibrated, ref_acc


def evaluate_classifier(clf: CalibratedClassifierCV,
                        labels: ImageLabels,
                        classes: List[int],
                        storage: Storage):
    """
    Return the accuracy of classifier "clf" evaluated on "imkeys"
    with ground truth given in "gtdict". Features are fetched from S3 "bucket".
    """
    scores, gts, ests = [], [], []
    for imkey in labels.image_keys:
        x, y = load_image_data(labels, imkey, classes, storage)
        if len(x) > 0:
            scores.extend(clf.predict_proba(x).max(axis=1))
            ests.extend(clf.predict(x))
            gts.extend(y)

    if len(gts) == 0:
        raise ValueError('Not enough data in validation set.')

    return gts, ests, scores


def chunkify(lst, nbr_chunks):
    return [lst[i::nbr_chunks] for i in range(nbr_chunks)]


def calc_batch_size(max_imgs_in_memory, train_set_size):
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
    assert len(image_labels) == len(image_features.point_features), \
        "number of extracted features doesn't match the number of labels!"

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
