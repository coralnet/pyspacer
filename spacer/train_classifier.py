import abc
import random
import json
import time
from typing import Tuple, List

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from spacer.storage import Storage

from spacer.messages import \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    ValResults, \
    FeatureLabels, \
    ImageFeatures


class ClassifierTrainer(abc.ABC):
    def __init__(self, msg: TrainClassifierMsg, storage: Storage):
        self.msg = msg
        self.storage = storage

    @abc.abstractmethod
    def __call__(self) -> Tuple[CalibratedClassifierCV,
                                ValResults,
                                TrainClassifierReturnMsg]:
        pass


class LinearTrainer(ClassifierTrainer):
    def __call__(self):

        t0 = time.time()

        # Load train labels
        train_labels = self.msg.load_train_feature_labels(self.storage)
        if len(train_labels) < 10:
            raise ValueError('Not enough training samples.')

        clf, ref_accs = train(train_labels, self.storage, self.msg.nbr_epochs)
        classes = list(clf.classes_)

        # Evaluate on val.
        val_labels = self.msg.load_val_feature_labels(self.storage)
        val_gts, val_ests, val_scores = evaluate_classifier(
            clf, val_labels, classes, self.storage)

        if len(val_gts) == 0:
            raise ValueError('Not enough data in validation set.')

        # Evaluate the previous classifiers on validation set.
        pc_accs = []
        for pc_model_key in self.msg.pc_models_key:
            this_clf = self.storage.load_classifier(pc_model_key)
            pc_gts, pc_ests, _ = evaluate_classifier(this_clf, val_labels,
                                                     classes, self.storage)
            pc_accs.append(calc_acc(pc_gts, pc_ests))

        return \
            clf, \
            TrainClassifierReturnMsg(
                acc=calc_acc(val_gts, val_ests),
                pc_accs=pc_accs,
                ref_accs=ref_accs,
                runtime=time.time() - t0
            ), ValResults(
                scores=val_scores,
                gt=[classes.index(member) for member in val_gts],
                est=[classes.index(member) for member in val_ests],
                classes=classes
            )


def train(feature_labels: FeatureLabels,
          storage: Storage,
          nbr_epochs: int) -> Tuple[CalibratedClassifierCV, List[float]]:
    # Calculate max nbr images to keep in memory (for 5000 samples total).
    max_imgs_in_memory = 5000 / feature_labels.samples_per_image

    # Make train and ref split.
    # Reference set is here a hold-out part of the train-data portion.
    # Purpose of reference set is to
    # 1) know accuracy per epoch
    # 2) calibrate classifier output scores.
    # We call it "reference" to disambiguate from the validation set.
    ref_set = feature_labels.image_keys[::10]
    random.shuffle(ref_set)
    ref_set = ref_set[:max_imgs_in_memory]  # Enforce memory limit.
    train_set = list(set(feature_labels.image_keys) - set(ref_set))
    print("-> Trainset: {}, valset: {} images".
          format(len(train_set), len(ref_set)))

    # Figure out # images per batch and batches per epoch.
    images_per_batch, batches_per_epoch = \
        calc_batch_size(max_imgs_in_memory, len(train_set))
    print("-> Using {} images per mini-batch and {} mini-batches per epoch".
          format(images_per_batch, batches_per_epoch))

    # Identify classes common to both train and val.
    # This will be our labelset for the training.
    trainclasses = feature_labels.unique_classes(train_set)
    refclasses = feature_labels.unique_classes(ref_set)
    classes = list(trainclasses.intersection(refclasses))
    print("-> Trainset: {}, valset: {}, common: {} labels".format(
        len(trainclasses), len(refclasses), len(classes)))
    if len(classes) == 1:
        raise ValueError('Not enough classes to do training (only 1)')

    # Load reference data (must hold in memory for the calibration)
    print("-> Loading reference data.")
    refx, refy = load_batch_data(feature_labels, ref_set, classes, storage)

    # Initialize classifier and ref set accuracy list
    print("-> Online training...")
    clf = SGDClassifier(loss='log', average=True)
    refacc = []
    for epoch in range(nbr_epochs):
        print("-> Epoch {}".format(epoch))
        random.shuffle(train_set)
        mini_batches = chunkify(train_set, batches_per_epoch)
        for mb in mini_batches:
            x, y = load_batch_data(feature_labels, mb, classes, storage)
            clf.partial_fit(x, y, classes=classes)
        refacc.append(calc_acc(refy, clf.predict(refx)))
        print("-> acc: {}".format(refacc[-1]))

    print("-> Calibrating.")
    clf_calibrated = CalibratedClassifierCV(clf, cv="prefit")
    clf_calibrated.fit(refx, refy)

    return clf_calibrated, refacc


def evaluate_classifier(clf: CalibratedClassifierCV,
                        feature_labels: FeatureLabels,
                        classes: List[int],
                        storage: Storage):
    """
    Return the accuracy of classifier "clf" evaluated on "imkeys"
    with ground truth given in "gtdict". Features are fetched from S3 "bucket".
    """
    scores, gt, est = [], [], []
    for imkey in feature_labels.image_keys:
        x, y = load_image_data(feature_labels, imkey, classes, storage)
        if len(x) > 0:
            scores.extend(np.max(clf.predict_proba(x)))
            est.extend(clf.predict(x))
            gt.extend(y)

    return gt, est, scores


def chunkify(lst, nbr_chunks):
    return [lst[i::nbr_chunks] for i in range(nbr_chunks)]


def calc_batch_size(max_imgs_in_memory, train_set_size):
    images_per_batch = min(max_imgs_in_memory, train_set_size)
    batches_per_epoch = int(np.ceil(train_set_size / images_per_batch))
    return images_per_batch, batches_per_epoch


def load_image_data(feature_labels: FeatureLabels,
                    imkey: str,
                    classes: List[int],
                    storage: Storage) -> Tuple[List[List[float]], List[int]]:
    """
    Loads features and labels for image and mathches feature with labels.
    """

    # Load features for this image.
    image_features = ImageFeatures.deserialize(
        json.loads(storage.load_string(imkey)))

    # Load row, col, labels for this image.
    image_labels = feature_labels.data[imkey]

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
        for _, _, label, point_feature in zip(image_labels,
                                              image_features.point_features):
            if label not in classes:
                continue
            x.append(point_feature.data)
            y.append(label)

    return x, y


def load_batch_data(feature_labels: FeatureLabels,
                    imkeylist: List[str],
                    classes: List[int],
                    storage: Storage) -> Tuple[List[List[float]], List[int]]:
    """ Loads features and labels and match them together. """
    x, y = [], []
    for imkey in imkeylist:
        x_, y_ = load_image_data(feature_labels, imkey, classes, storage)
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
        if not isinstance(g, int):
            raise TypeError('Input gt must be an array of ints')

    for e in est:
        if not isinstance(e, int):
            raise TypeError('Input est must be an array of ints')

    return float(sum([(g == e) for (g, e) in zip(gt, est)])) / len(gt)


def trainer_factory(msg: TrainClassifierMsg,
                    storage: Storage) -> ClassifierTrainer:
    """ There is only one type of Trainer, so this factory is trivial. """
    return LinearTrainer(msg=msg, storage=storage)
