"""
Defines train-classifier ABC; implementations; and factory.
"""

from __future__ import annotations
import abc
import inspect
import time
from importlib import import_module
from logging import getLogger

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from spacer import config
from spacer.data_classes import ValResults
from spacer.messages import TrainClassifierReturnMsg, TrainingTaskLabels
from spacer.train_utils import evaluate_classifier, calc_acc

logger = getLogger(__name__)

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
    to train the classifier.
    """

    def __init__(
        self,
        batch_size: int = config.TRAINING_BATCH_LABEL_COUNT,
        # MLP settings — None means auto-select based on label count
        mlp_hidden_layer_sizes: tuple[int, ...] | None = None,
        mlp_learning_rate_init: float | None = None,
        # MLP auto-select threshold
        mlp_large_data_threshold: int = 50000,
        # SGD settings
        sgd_loss: str = 'log_loss',
        sgd_average: bool = True,
        sgd_random_state: int = 0,
    ):
        self.batch_size = batch_size
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.mlp_learning_rate_init = mlp_learning_rate_init
        self.mlp_large_data_threshold = mlp_large_data_threshold
        self.sgd_loss = sgd_loss
        self.sgd_average = sgd_average
        self.sgd_random_state = sgd_random_state

    def serialize(self) -> dict:
        data = super().serialize()
        sig = inspect.signature(self.__init__)
        for name, param in sig.parameters.items():
            value = getattr(self, name)
            if value != param.default:
                data[name] = value
        return data

    def _train(self, train_labels, ref_labels, nbr_epochs, clf_type):
        logger.debug(
            f"Data sets:"
            f" Train = {len(train_labels)} images,"
            f" {train_labels.label_count} labels;"
            f" Ref = {len(ref_labels)} images,"
            f" {ref_labels.label_count} labels")
        logger.debug(
            f"Mini-batch size: {self.batch_size} labels")

        classes_list = list(ref_labels.classes_set)

        with config.log_entry_and_exit("loading of reference data"):
            refx, refy = ref_labels.load_all_data()

        with config.log_entry_and_exit("training using " + clf_type):
            if clf_type == 'MLP':
                if self.mlp_hidden_layer_sizes is not None:
                    hls = self.mlp_hidden_layer_sizes
                    lr = self.mlp_learning_rate_init or 1e-3
                elif train_labels.label_count >= self.mlp_large_data_threshold:
                    hls, lr = (200, 100), 1e-4
                else:
                    hls, lr = (100,), 1e-3
                clf = MLPClassifier(
                    hidden_layer_sizes=hls, learning_rate_init=lr)
            else:
                clf = SGDClassifier(
                    loss=self.sgd_loss,
                    average=self.sgd_average,
                    random_state=self.sgd_random_state,
                )

            ref_acc = []

            for epoch in range(nbr_epochs):
                for x, y in train_labels.load_data_in_batches(
                    batch_size=self.batch_size,
                    random_seed=epoch,
                ):
                    clf.partial_fit(x, y, classes=classes_list)

                ref_acc.append(calc_acc(refy, clf.predict(refx)))
                logger.debug(f"Epoch {epoch}, acc: {ref_acc[-1]}")

        with config.log_entry_and_exit("calibration"):
            clf_calibrated = CalibratedClassifierCV(clf, cv="prefit")
            clf_calibrated.fit(refx, refy)

        return clf_calibrated, ref_acc

    def __call__(self,
                 labels,
                 nbr_epochs,
                 pc_models,
                 clf_type):

        assert clf_type in config.CLASSIFIER_TYPES
        # Train.
        t0 = time.time()
        clf, ref_accs = self._train(
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
