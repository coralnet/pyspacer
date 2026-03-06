"""
Defines message types as data-classes.
Each data-class can serialize itself to a structure of JSON-friendly
python-native data-structures such that it can be stored.
"""

from __future__ import annotations
import dataclasses
from pathlib import Path

from spacer import config
from spacer.data_classes import (
    DataClass, DataLocation, ImageLabels, LabelId)


class ExtractFeaturesMsg(DataClass):
    """ Input message for extract-features task. """

    def __init__(self,
                 job_token: str,  # Token for caller DB reference.
                 extractor: 'FeatureExtractor',
                 rowcols: list[tuple[int, int]],  # List of (row, col) entries.
                 image_loc: DataLocation,  # Where to fetch image.
                 feature_loc: DataLocation,  # Where to store output.
                 ):

        assert isinstance(rowcols, list)
        assert len(rowcols) > 0, "Invalid message, rowcols entry is empty."
        assert len(rowcols[0]) == 2
        assert feature_loc.is_writable, (
            f"Write not supported for"
            f" '{feature_loc.storage_type}' storage type.")

        self.job_token = job_token
        self.extractor = extractor
        self.rowcols = rowcols
        self.image_loc = image_loc
        self.feature_loc = feature_loc

    @classmethod
    def example(cls) -> 'ExtractFeaturesMsg':
        from spacer.extractors import EfficientNetExtractor
        return ExtractFeaturesMsg(
            job_token='123abc',
            extractor=EfficientNetExtractor(
                data_locations=dict(
                    weights=DataLocation('filesystem', '/path/to/weights.pt'),
                )
            ),
            rowcols=[(100, 100)],
            image_loc=DataLocation('memory', 'my_image.jpg'),
            feature_loc=DataLocation('memory', 'my_feats.json'),
        )

    def serialize(self) -> dict:
        return {
            'job_token': self.job_token,
            'extractor': self.extractor.serialize(),
            'rowcols': self.rowcols,
            'image_loc': self.image_loc.serialize(),
            'feature_loc': self.feature_loc.serialize()
        }

    @classmethod
    def deserialize(cls, data: dict) -> 'ExtractFeaturesMsg':
        from spacer.extractors import FeatureExtractor
        return ExtractFeaturesMsg(
            job_token=data['job_token'],
            extractor=FeatureExtractor.deserialize(data['extractor']),
            rowcols=[tuple(rc) for rc in data['rowcols']],
            image_loc=DataLocation.deserialize(data['image_loc']),
            feature_loc=DataLocation.deserialize(data['feature_loc'])
        )


class ExtractFeaturesReturnMsg(DataClass):
    """ Return message for extract_features task. """

    def __init__(self,
                 extractor_loaded_remotely: bool,
                 runtime: float):

        self.extractor_loaded_remotely = extractor_loaded_remotely
        self.runtime = runtime

    @classmethod
    def example(cls) -> 'ExtractFeaturesReturnMsg':
        return ExtractFeaturesReturnMsg(
            extractor_loaded_remotely=True,
            runtime=2.1
        )


@dataclasses.dataclass
class TrainingTaskLabels:
    """
    This structure specifies sets of point-locations to ground-truth-labels
    (annotations) mappings to use for classifier training. Three data sets
    are included: training, reference, and validation.

    For ease of use, the function preprocess_labels() can help build
    this from a single ImageLabels instance.

    The reference set is meant to fit in a single
    training batch, so the label count should be no greater than
    config.TRAINING_BATCH_LABEL_COUNT. This isn't strictly
    enforced, but helps ensure stabler performance/results.

    The training set is what the classifier actually learns from.
    The reference set is a hold-out set whose purpose is to
    1) get an accuracy measurement per epoch, and
    2) calibrate classifier output scores.
    The validation set is used to evaluate accuracy of the final classifier.
    """

    # Since this is a Python dataclass, there's an automatically generated
    # __init__() method which sets the fields below.
    train: ImageLabels
    ref: ImageLabels
    val: ImageLabels

    def __getitem__(self, item):
        """
        Can access a particular data set with either obj.<setname>,
        or obj['<setname>'].
        """
        if item not in ['train', 'ref', 'val']:
            raise KeyError
        return getattr(self, item)

    def __setitem__(self, item, value):
        if item not in ['train', 'ref', 'val']:
            raise KeyError
        setattr(self, item, value)

    @property
    def label_count(self):
        return (
            self.train.label_count
            + self.ref.label_count
            + self.val.label_count)

    @property
    def has_remote_data(self) -> bool:
        return (
            self.train.has_remote_data
            or self.ref.has_remote_data
            or self.val.has_remote_data)

    def set_filesystem_cache(self, dir_path: str):
        self.train.set_filesystem_cache(dir_path)
        self.ref.set_filesystem_cache(dir_path)
        self.val.set_filesystem_cache(dir_path)

    def serialize(self) -> dict:
        return {
            'train': self.train.serialize(),
            'ref': self.ref.serialize(),
            'val': self.val.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> 'TrainingTaskLabels':
        return TrainingTaskLabels(
            train=ImageLabels.deserialize(data['train']),
            ref=ImageLabels.deserialize(data['ref']),
            val=ImageLabels.deserialize(data['val']),
        )

    @classmethod
    def example(cls) -> 'TrainingTaskLabels':
        return TrainingTaskLabels(
            train=ImageLabels.example(),
            ref=ImageLabels.example(),
            val=ImageLabels.example(),
        )


class TrainClassifierMsg(DataClass):
    """ Specifies the train classifier task. """

    # When on Python 3.11+ only, we could define this as a StrEnum to be more
    # idiomatic.
    class FeatureCache:
        AUTO = 'auto'
        DISABLED = 'disabled'

    def __init__(
        self,
        # For caller's reference.
        job_token: str,
        # A ClassifierTrainer instance, or a string that resolves to one
        # via trainer_factory() (e.g. 'minibatch' or a fully-qualified
        # class path like 'mypackage.trainers.MyTrainer').
        trainer: 'ClassifierTrainer | str',
        # How many iterations the training algorithm should run; more epochs
        # = more opportunity to converge to a better fit, but slower.
        nbr_epochs: int,
        # Classifier types available:
        # 1. 'MLP': multi-layer perceptron; newer classifier type for CoralNet
        # 2. 'LR': logistic regression; older classifier type for CoralNet
        clf_type: str,
        # Point-locations to ground-truth-labels (annotations) mappings
        # used to train the classifier.
        # See TrainingTaskLabels comments for more info.
        labels: TrainingTaskLabels,
        # List of previously-created models (classifiers) to also evaluate
        # using this dataset, for informational purposes only.
        # A classifier is stored as a pickled CalibratedClassifierCV.
        previous_model_locs: list[DataLocation],
        # Where the new model (classifier) should be output to.
        model_loc: DataLocation,
        # Where the detailed evaluation results of the new model should be
        # stored.
        valresult_loc: DataLocation,
        # If feature vectors are loaded from remote storage, this specifies
        # where the feature-vector cache (a temporary directory in the local
        # filesystem) is located. Can be:
        # - The special value FeatureCache.AUTO, which lets the OS decide where
        #   the temporary directory lives. (Default)
        # - The special value FeatureCache.DISABLED, which makes feature
        #   vectors get loaded remotely every time without being cached
        #   (which means most vectors will be remote-loaded once per epoch).
        #   This would be desired if there isn't enough disk space to cache all
        #   features.
        # - Absolute path to the directory where the cache will live, either
        #   as a str or a pathlib.Path.
        feature_cache_dir: str | Path = FeatureCache.AUTO,
    ):

        if isinstance(trainer, str):
            from spacer.train_classifier import trainer_factory
            trainer = trainer_factory(trainer)

        self.job_token = job_token
        self.trainer = trainer
        self.nbr_epochs = nbr_epochs
        self.clf_type = clf_type
        self.labels = labels
        self.previous_model_locs = previous_model_locs
        self.model_loc = model_loc
        self.valresult_loc = valresult_loc
        self.feature_cache_dir = feature_cache_dir

    @classmethod
    def example(cls):
        return TrainClassifierMsg(
            job_token='123_abc',
            trainer='minibatch',
            nbr_epochs=2,
            clf_type='MLP',
            labels=TrainingTaskLabels.example(),
            previous_model_locs=[
                DataLocation('memory', 'previous_model1.pkl'),
                DataLocation('memory', 'previous_model2.pkl'),
            ],
            model_loc=DataLocation('memory', 'my_new_model.pkl'),
            valresult_loc=DataLocation('memory', 'my_valresult.json'),
        )

    def serialize(self) -> dict:
        return {
            'job_token': self.job_token,
            'trainer': self.trainer.serialize(),
            'nbr_epochs': self.nbr_epochs,
            'clf_type': self.clf_type,
            'labels': self.labels.serialize(),
            'previous_model_locs': [entry.serialize()
                                    for entry in self.previous_model_locs],
            'model_loc': self.model_loc.serialize(),
            'valresult_loc': self.valresult_loc.serialize(),
            'feature_cache_dir': str(self.feature_cache_dir),
        }

    @classmethod
    def deserialize(cls, data: dict) -> 'TrainClassifierMsg':
        # Support legacy format with 'trainer_name' key.
        if 'trainer_name' in data:
            from spacer.train_classifier import trainer_factory
            trainer = trainer_factory(data['trainer_name'])
        else:
            from spacer.train_classifier import ClassifierTrainer
            trainer = ClassifierTrainer.deserialize(data['trainer'])

        return TrainClassifierMsg(
            job_token=data['job_token'],
            trainer=trainer,
            nbr_epochs=data['nbr_epochs'],
            clf_type=data['clf_type'],
            labels=TrainingTaskLabels.deserialize(data['labels']),
            previous_model_locs=[DataLocation.deserialize(entry)
                                 for entry in data['previous_model_locs']],
            model_loc=DataLocation.deserialize(data['model_loc']),
            valresult_loc=DataLocation.deserialize(data['valresult_loc']),
            feature_cache_dir=data['feature_cache_dir'],
        )


class TrainClassifierReturnMsg(DataClass):
    """ Return message for train_classifier task. """

    def __init__(self,
                 # Accuracy of new classifier on the validation set.
                 acc: float,
                 # Accuracy of previous classifiers on the validation set.
                 pc_accs: list[float],
                 # Accuracy on reference set for each epoch of training.
                 ref_accs: list[float],
                 # Runtime for full training execution.
                 runtime: float,
                 ):
        self.acc = acc
        self.pc_accs = pc_accs
        self.ref_accs = ref_accs
        self.runtime = runtime

    @classmethod
    def example(cls):
        return TrainClassifierReturnMsg(
            acc=0.7,
            pc_accs=[0.4, 0.5, 0.6],
            ref_accs=[0.55, 0.65, 0.64, 0.67, 0.70],
            runtime=123.4,
        )

    @classmethod
    def deserialize(cls, data: dict) -> 'TrainClassifierReturnMsg':
        return TrainClassifierReturnMsg(**data)


class ClassifyFeaturesMsg(DataClass):
    """ Specifies the classify_features task. """

    def __init__(self,
                 job_token: str,
                 feature_loc: DataLocation,
                 classifier_loc: DataLocation):
        self.job_token = job_token
        self.feature_loc = feature_loc
        self.classifier_loc = classifier_loc

    @classmethod
    def example(cls):
        return ClassifyFeaturesMsg(
            job_token='my_job',
            feature_loc=DataLocation(storage_type='url',
                                     key='https://my-bucket.s3-my-region.'
                                         'amazonaws.com/01234aeiou.png.'
                                         'featurevector'),
            classifier_loc=DataLocation(storage_type='url',
                                        key='https://my-bucket.s3-my-region.'
                                        'amazonaws.com/my_model_id.model')
        )

    def serialize(self):
        return {
            'job_token': self.job_token,
            'feature_loc': self.feature_loc.serialize(),
            'classifier_loc': self.classifier_loc.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> 'ClassifyFeaturesMsg':
        return ClassifyFeaturesMsg(
            job_token=data['job_token'],
            feature_loc=DataLocation.deserialize(data['feature_loc']),
            classifier_loc=DataLocation.deserialize(data['classifier_loc'])
        )


class ClassifyImageMsg(DataClass):
    """ Specifies the classify_image task. """

    def __init__(self,
                 job_token: str,  # Primary key of job, not used in spacer.
                 extractor: 'FeatureExtractor',
                 rowcols: list[tuple[int, int]],
                 image_loc: DataLocation,  # Location of image to classify.
                 classifier_loc: DataLocation,  # Location of classifier.
                 ):
        self.job_token = job_token
        self.extractor = extractor
        self.rowcols = rowcols
        self.image_loc = image_loc
        self.classifier_loc = classifier_loc

    @classmethod
    def example(cls):
        from spacer.extractors import EfficientNetExtractor
        return ClassifyImageMsg(
            job_token='my_job',
            extractor=EfficientNetExtractor(
                data_locations=dict(
                    weights=DataLocation('filesystem', '/path/to/weights.pt'),
                )
            ),
            rowcols=[(1, 1), (2, 2)],
            image_loc=DataLocation(storage_type='url',
                                   key='https://my-bucket.s3-my-region.'
                                   'amazonaws.com/01234aeiou.png'),
            classifier_loc=DataLocation(storage_type='url',
                                        key='https://my-bucket.s3-my-region.'
                                        'amazonaws.com/my_model_id.model')
        )

    def serialize(self):
        return {
            'job_token': self.job_token,
            'extractor': self.extractor.serialize(),
            'rowcols': self.rowcols,
            'image_loc': self.image_loc.serialize(),
            'classifier_loc': self.classifier_loc.serialize()
        }

    @classmethod
    def deserialize(cls, data: dict) -> 'ClassifyImageMsg':
        from spacer.extractors import FeatureExtractor
        return ClassifyImageMsg(
            job_token=data['job_token'],
            extractor=FeatureExtractor.deserialize(data['extractor']),
            rowcols=[tuple(rc) for rc in data['rowcols']],
            image_loc=DataLocation.deserialize(data['image_loc']),
            classifier_loc=DataLocation.deserialize(data['classifier_loc'])
        )


class ClassifyReturnMsg(DataClass):
    """ Return message from the classify_{image, features} tasks. """

    def __init__(self,
                 runtime: float,
                 # Scores is a list of (row, col, [scores]) tuples.
                 scores: list[tuple[int, int, list[float]]],
                 # Maps the score index to a global class id.
                 classes: list[LabelId],
                 valid_rowcol: bool):

        self.runtime = runtime
        self.scores = scores
        self.classes = classes
        self.valid_rowcol = valid_rowcol

    def __getitem__(self, rowcol: tuple[int, int]) -> list[float]:
        """ Returns features at (row, col) location. """
        if not self.valid_rowcol:
            raise ValueError('Method requires valid rows and columns')
        rc_set = {(row, col): scores for row, col, scores in self.scores}
        return rc_set[rowcol]

    @classmethod
    def example(cls):
        return ClassifyReturnMsg(
            runtime=1.1,
            scores=[(10, 20, [0.1, 0.2, 0.7]), (20, 40, [0.9, 0.06, 0.04])],
            classes=[100, 12, 44],
            valid_rowcol=True
        )

    @classmethod
    def deserialize(cls, data: dict) -> 'ClassifyReturnMsg':
        return ClassifyReturnMsg(
            runtime=data['runtime'],
            scores=[(row, col, scores) for
                    row, col, scores in data['scores']],
            classes=data['classes'],
            valid_rowcol=data['valid_rowcol']
        )


class JobMsg(DataClass):
    """ Highest level message which hold task messages.
    A job can contain multiple tasks.
    """

    def __init__(self,
                 task_name: str,
                 tasks: list[ExtractFeaturesMsg
                             | TrainClassifierMsg
                             | ClassifyFeaturesMsg
                             | ClassifyImageMsg]):

        assert task_name in config.TASKS

        self.task_name = task_name
        self.tasks = tasks

    @classmethod
    def deserialize(cls, data: dict) -> 'JobMsg':

        task_name = data['task_name']
        assert task_name in config.TASKS
        if task_name == 'extract_features':
            deserializer = ExtractFeaturesMsg.deserialize
        elif task_name == 'train_classifier':
            deserializer = TrainClassifierMsg.deserialize
        elif task_name == 'classify_features':
            deserializer = ClassifyFeaturesMsg.deserialize
        else:
            deserializer = ClassifyImageMsg.deserialize

        return JobMsg(task_name, [deserializer(item) for
                                  item in data['tasks']])

    def serialize(self):
        return {
            'task_name': self.task_name,
            'tasks': [job.serialize() for job in self.tasks]
        }

    @classmethod
    def example(cls):
        return JobMsg(task_name='classify_image',
                      tasks=[ClassifyImageMsg.example()])


class JobReturnMsg(DataClass):
    """ Highest level return message. """

    def __init__(self,
                 original_job: JobMsg,
                 ok: bool,
                 results: list[ExtractFeaturesReturnMsg
                               | TrainClassifierReturnMsg
                               | ClassifyReturnMsg] | None,
                 error_message: str | None):

        self.original_job = original_job
        self.results = results
        self.ok = ok
        self.error_message = error_message

    @classmethod
    def example(cls):
        return JobReturnMsg(
            original_job=JobMsg.example(),
            ok=True,
            results=[ClassifyReturnMsg.example()],
            error_message=None
        )

    @classmethod
    def deserialize(cls, data: dict) -> 'JobReturnMsg':

        original_job = JobMsg.deserialize(data['original_job'])

        if data['ok']:
            task_name = original_job.task_name
            assert task_name in config.TASKS
            if task_name == 'extract_features':
                deserializer = ExtractFeaturesReturnMsg.deserialize
            elif task_name == 'train_classifier':
                deserializer = TrainClassifierReturnMsg.deserialize
            else:  # task_name in ['classify_image', 'classify_features']
                deserializer = ClassifyReturnMsg.deserialize
            results = [deserializer(task_res) for task_res in data['results']]
        else:
            results = data['results']

        return JobReturnMsg(
            original_job=original_job,
            ok=data['ok'],
            results=results,
            error_message=data['error_message']
        )

    def serialize(self):
        if self.ok:
            results = [task_res.serialize() for task_res in self.results]
        else:
            results = self.results
        return {
            'original_job': self.original_job.serialize(),
            'ok': self.ok,
            'results': results,
            'error_message': self.error_message
        }
