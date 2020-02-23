import abc
import time
from typing import Tuple

from sklearn.calibration import CalibratedClassifierCV

from spacer.messages import \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    ValResults
from spacer.storage import Storage
from spacer.train_utils import train, evaluate_classifier, calc_acc


class ClassifierTrainer(abc.ABC):
    def __init__(self, msg: TrainClassifierMsg, storage: Storage):
        self.msg = msg
        self.storage = storage

    @abc.abstractmethod
    def __call__(self) -> Tuple[CalibratedClassifierCV,
                                ValResults,
                                TrainClassifierReturnMsg]:
        pass


class DefaultTrainer(ClassifierTrainer):
    """ This is the default trainer. """

    def __call__(self):

        # Train.
        t0 = time.time()
        train_labels = self.msg.load_train_feature_labels(self.storage)
        clf, ref_accs = train(train_labels, self.storage, self.msg.nbr_epochs)
        classes = list(clf.classes_)

        # Evaluate new classifier on validation set.
        val_labels = self.msg.load_val_feature_labels(self.storage)
        val_gts, val_ests, val_scores = evaluate_classifier(
            clf, val_labels, classes, self.storage)

        # Evaluate previous classifiers on validation set.
        pc_accs = []
        for pc_model_key in self.msg.pc_models_key:
            this_clf = self.storage.load_classifier(pc_model_key)
            pc_gts, pc_ests, _ = evaluate_classifier(this_clf, val_labels,
                                                     classes, self.storage)
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


def trainer_factory(msg: TrainClassifierMsg,
                    storage: Storage) -> ClassifierTrainer:
    """ There is only one type of Trainer, so this factory is trivial. """
    return DefaultTrainer(msg=msg, storage=storage)
