import abc
import json
import time
from collections import defaultdict
from typing import Tuple, List

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from spacer import config
from spacer.data_classes import ImageLabels, ValResults
from spacer.messages import \
    TrainClassifierReturnMsg
from spacer.storage import Storage
from spacer.train_utils import train, evaluate_classifier, calc_acc, \
    make_random_data


class ClassifierTrainer(abc.ABC):  # pragma: no cover

    @abc.abstractmethod
    def __call__(self,
                 traindata_key: str,
                 valdata_key: str,
                 nbr_epochs: int,
                 pc_models_key: List[str],
                 storage: Storage) \
            -> Tuple[CalibratedClassifierCV,
                     ValResults,
                     TrainClassifierReturnMsg]:
        pass


class DummyTrainer(ClassifierTrainer):
    """ Returns a classifier trained on made-up dummy-data """

    def __init__(self,
                 feature_dim: int = 12,
                 class_list: List[int] = None,
                 points_per_image: int = 10,
                 n_valdata: int = 10,
                 n_traindata: int = 100
                 ):
        if class_list is None:
            class_list = [1, 2, 3]
        self.feature_dim = feature_dim
        self.class_list = class_list
        self.points_per_image = points_per_image
        self.n_valdata = n_valdata
        self.n_traindata = n_traindata

    def __call__(self,
                 traindata_key,
                 valdata_key,
                 nbr_epochs,
                 pc_models_key,
                 storage):

        t0 = time.time()
        np.random.seed(0)

        labels = make_random_data(self.n_traindata, self.class_list,
                                  self.points_per_image, self.feature_dim,
                                  storage)

        # Call the train routine on dummy data to make sure the classifier
        # is trained and calibrated (so that it won't return NaNs when called).
        clf, ref_accs = train(labels, storage, nbr_epochs)

        nbr_val_pts = self.n_valdata*self.points_per_image
        val_gts = np.random.choice(len(self.class_list), nbr_val_pts).tolist()
        val_ests = np.random.choice(len(self.class_list), nbr_val_pts).tolist()

        return \
            clf, \
            ValResults(
                scores=np.random.random(nbr_val_pts).tolist(),
                gt=val_gts,
                est=val_ests,
                classes=self.class_list
            ), \
            TrainClassifierReturnMsg(
                acc=calc_acc(val_gts, val_ests),
                pc_accs=np.random.random(len(pc_models_key)).tolist(),
                ref_accs=ref_accs,
                runtime=time.time() - t0
            )


class MiniBatchTrainer(ClassifierTrainer):
    """
    This is the default trainer. It uses mini-batches of data
    to train the classifier
    """

    def __call__(self,
                 traindata_key,
                 valdata_key,
                 nbr_epochs,
                 pc_models_key,
                 storage):

        # Train.
        t0 = time.time()
        train_labels = ImageLabels.deserialize(
            json.loads(storage.load_string(traindata_key)))
        clf, ref_accs = train(train_labels, storage, nbr_epochs)
        classes = list(clf.classes_)

        # Evaluate new classifier on validation set.
        val_labels = ImageLabels.deserialize(
            json.loads(storage.load_string(valdata_key)))
        val_gts, val_ests, val_scores = evaluate_classifier(
            clf, val_labels, classes, storage)

        # Evaluate previous classifiers on validation set.
        pc_accs = []
        for pc_model_key in pc_models_key:
            this_clf = storage.load_classifier(pc_model_key)
            pc_gts, pc_ests, _ = evaluate_classifier(this_clf, val_labels,
                                                     classes, storage)
            pc_accs.append(calc_acc(pc_gts, pc_ests))

        return \
            clf, \
            ValResults(
                scores=val_scores,
                gt=[classes.index(member) for member in val_gts],
                est=[classes.index(member) for member in val_ests],
                classes=classes
            ), \
            TrainClassifierReturnMsg(
                acc=calc_acc(val_gts, val_ests),
                pc_accs=pc_accs,
                ref_accs=ref_accs,
                runtime=time.time() - t0
            )


def trainer_factory(trainer_name: str, dummy_kwargs=defaultdict(str)) \
        -> ClassifierTrainer:
    """ There is only one type of Trainer, so this factory is trivial. """
    assert trainer_name in config.TRAINER_NAMES
    if trainer_name == 'minibatch':
        return MiniBatchTrainer()
    if trainer_name == 'dummy':
        return DummyTrainer(**dummy_kwargs)
