"""
Defines train-classifier ABC; implementations; and factory.
"""

from __future__ import annotations
import abc
import time

from sklearn.calibration import CalibratedClassifierCV

from spacer import config
from spacer.data_classes import ImageLabels, ValResults
from spacer.messages import TrainClassifierReturnMsg, DataLocation
from spacer.train_utils import train, evaluate_classifier, calc_acc


class ClassifierTrainer(abc.ABC):  # pragma: no cover

    @abc.abstractmethod
    def __call__(self,
                 train_labels: ImageLabels,
                 val_labels: ImageLabels,
                 nbr_epochs: int,
                 pc_models: list[CalibratedClassifierCV],
                 feature_loc: DataLocation,
                 clf_type: str) \
            -> tuple[CalibratedClassifierCV,
                     ValResults,
                     TrainClassifierReturnMsg]:
        pass


class MiniBatchTrainer(ClassifierTrainer):
    """
    This is the default trainer. It uses mini-batches of data
    to train the classifier
    """

    def __call__(self,
                 train_labels,
                 val_labels,
                 nbr_epochs,
                 pc_models,
                 feature_loc,
                 clf_type):

        assert clf_type in config.CLASSIFIER_TYPES
        # Train.
        t0 = time.time()
        clf, ref_accs = train(train_labels, feature_loc, nbr_epochs, clf_type)
        classes = clf.classes_.tolist()

        # Evaluate new classifier on validation set.
        val_gts, val_ests, val_scores = evaluate_classifier(
            clf, val_labels, classes, feature_loc)

        # Evaluate previous classifiers on validation set.
        pc_accs = []
        for pc_model in pc_models:
            pc_gts, pc_ests, _ = evaluate_classifier(pc_model, val_labels,
                                                     classes, feature_loc)
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


def trainer_factory(trainer_name: str) -> ClassifierTrainer:
    """ There is only one type of Trainer, so this factory is trivial. """
    assert trainer_name in config.TRAINER_NAMES
    if trainer_name == 'minibatch':
        return MiniBatchTrainer()
