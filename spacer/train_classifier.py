"""
Defines train-classifier ABC; implementations; and factory.
"""

from __future__ import annotations
import abc
import time
from importlib import import_module

from sklearn.calibration import CalibratedClassifierCV

from spacer import config
from spacer.data_classes import ValResults
from spacer.messages import TrainClassifierReturnMsg, TrainingTaskLabels
from spacer.train_utils import train, evaluate_classifier, calc_acc

_BUILTIN_TRAINERS: dict[str, str] = {
    'minibatch': 'spacer.train_classifier.MiniBatchTrainer',
}


class ClassifierTrainer(abc.ABC):  # pragma: no cover

    @abc.abstractmethod
    def __call__(self,
                 labels: TrainingTaskLabels,
                 nbr_epochs: int,
                 pc_models: list[CalibratedClassifierCV],
                 clf_type: str) \
            -> tuple[CalibratedClassifierCV,
                     ValResults,
                     TrainClassifierReturnMsg]:
        pass

    def serialize(self) -> dict:
        cls = self.__class__
        return dict(
            class_path=f'{cls.__module__}.{cls.__name__}',
        )

    @staticmethod
    def deserialize(data: dict) -> 'ClassifierTrainer':
        working_data = data.copy()
        class_path = working_data.pop('class_path')

        module_path, class_name = class_path.rsplit('.', 1)
        module = import_module(module_path)
        trainer_class = getattr(module, class_name)

        if not (
            isinstance(trainer_class, type)
            and issubclass(trainer_class, ClassifierTrainer)
        ):
            raise TypeError(
                f"{class_path} is not a ClassifierTrainer subclass"
            )

        return trainer_class(**working_data)

    def __repr__(self):
        return str(self.serialize())

    def __eq__(self, other):
        return self.serialize() == other.serialize()


class MiniBatchTrainer(ClassifierTrainer):
    """
    This is the default trainer. It uses mini-batches of data
    to train the classifier
    """

    def __call__(self,
                 labels,
                 nbr_epochs,
                 pc_models,
                 clf_type):

        assert clf_type in config.CLASSIFIER_TYPES
        # Train.
        t0 = time.time()
        clf, ref_accs = train(
            labels['train'], labels['ref'], nbr_epochs, clf_type)
        classes = clf.classes_.tolist()

        # Evaluate new classifier on validation set.
        val_gts, val_ests, val_scores = evaluate_classifier(
            clf, labels['val'])

        # Evaluate previous classifiers on validation set.
        pc_accs = []
        for pc_model in pc_models:
            pc_gts, pc_ests, _ = evaluate_classifier(
                pc_model, labels['val'])
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
    """
    Resolve a trainer name or class path to a ClassifierTrainer instance.

    Accepts either a built-in alias (e.g. 'minibatch') or a fully-qualified
    class path (e.g. 'mypackage.trainers.MyTrainer').
    """
    class_path = _BUILTIN_TRAINERS.get(trainer_name, trainer_name)
    return ClassifierTrainer.deserialize({'class_path': class_path})
