"""
Defines message types as data-classes.
Each data-class can serialize itself to a structure of JSON-friendly
python-native data-structures such that it can be stored.
"""

from typing import List, Tuple, Dict, Union, Optional

from spacer import config
from spacer.data_classes import DataClass


class ExtractFeaturesMsg(DataClass):
    """ Input message for extract features class. """

    def __init__(self,
                 pk: int,  # Primary key, for caller DB reference.
                 feature_extractor_name: str,  # Which extractor to use.
                 bucketname: str,  # If storage_type=s3, which bucket to use?
                 imkey: str,  # key or path to image to process.
                 rowcols: List[Tuple[int, int]],  # List of [row, col] entries.
                 outputkey: str,  # key or path to where to store output.
                 storage_type: str = 's3',  # type of storage.
                 ):

        assert storage_type in config.STORAGE_TYPES
        assert feature_extractor_name in config.FEATURE_EXTRACTOR_NAMES
        assert isinstance(rowcols, List)
        assert len(rowcols) > 0, "Invalid message, rowcols entry is empty."
        assert len(rowcols[0]) == 2

        self.pk = pk
        self.feature_extractor_name = feature_extractor_name
        self.bucketname = bucketname
        self.imkey = imkey
        self.storage_type = storage_type
        self.rowcols = rowcols
        self.outputkey = outputkey

    @classmethod
    def deserialize(cls, data: Dict) -> 'ExtractFeaturesMsg':
        """ Custom deserializer required to convert back to tuples. """
        msg = cls(**data)
        msg.rowcols = [tuple(rc) for rc in data['rowcols']]
        return msg

    @classmethod
    def example(cls) -> 'ExtractFeaturesMsg':
        return ExtractFeaturesMsg(
            pk=1,
            feature_extractor_name='vgg16_coralnet_ver1',
            bucketname='spacer-test',
            imkey='edinburgh3.jpg',
            rowcols=[(100, 100)],
            outputkey='edinburgh3.jpg.feats',
            storage_type='s3',
        )


class ExtractFeaturesReturnMsg(DataClass):
    """ Return message for extract_features task. """

    def __init__(self,
                 model_was_cashed: bool,
                 runtime: float):

        self.model_was_cashed = model_was_cashed
        self.runtime = runtime

    @classmethod
    def example(cls) -> 'ExtractFeaturesReturnMsg':
        return ExtractFeaturesReturnMsg(
            model_was_cashed=True,
            runtime=2.1
        )


class TrainClassifierMsg(DataClass):
    """ Specifies the train classifier task. """

    def __init__(self,
                 # Primary key of the model to train. Not used in spacer.
                 pk: int,
                 # Key for where to store the trained model.
                 model_key: str,
                 # Name of trainer to use.
                 trainer_name: str,
                 # Key to ImageLabels structure with training data.
                 traindata_key: str,
                 # Key to ImageLabels structure with validation data.
                 valdata_key: str,
                 # Key to where the validation results is stored.
                 valresult_key: str,
                 # Number of epochs to do training.
                 nbr_epochs: int,
                 # List of keys to to previous models
                 pc_models_key: List[str],
                 # List of primary-keys to previous models. This is for
                 # bookkeeping purposes.
                 pc_pks: List[int],
                 # Bucket name where features are stored.
                 bucketname: str,
                 # storage type
                 storage_type: str = 's3',
                 ):

        assert trainer_name in config.TRAINER_NAMES

        self.pk = pk
        self.model_key = model_key
        self.trainer_name = trainer_name
        self.traindata_key = traindata_key
        self.valdata_key = valdata_key
        self.valresult_key = valresult_key
        self.nbr_epochs = nbr_epochs
        self.pc_models_key = pc_models_key
        self.pc_pks = pc_pks
        self.bucketname = bucketname
        self.storage_type = storage_type

    @classmethod
    def example(cls):
        return TrainClassifierMsg(
            pk=1,
            model_key='my_trained_model',
            trainer_name='minibatch',
            traindata_key='my_traindata',
            valdata_key='my_valdata',
            valresult_key='my_valresults',
            nbr_epochs=5,
            pc_models_key=['my_previous_model1',
                           'my_previous_model2',
                           'my_previous_model3'],
            pc_pks=[1, 2, 3],
            bucketname='spacer-test',
        )

    @classmethod
    def deserialize(cls, data: Dict) -> 'TrainClassifierMsg':
        """ Redefining to help pycharm typing module """
        return cls(**data)


class TrainClassifierReturnMsg(DataClass):
    """ Return message for train_classifier task. """

    def __init__(self,
                 # Accuracy of new classifier on the validation set.
                 acc: float,
                 # Accuracy of previous classifiers on the validation set.
                 pc_accs: List[float],
                 # Accuracy on reference set for each epoch of training.
                 ref_accs: List[float],
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


class DeployMsg(DataClass):
    """ Specifies the deploy task. """

    def __init__(self,
                 pk: int,  # Primary key of job, not used in spacer.
                 im_url: str,  # URL of image to deploy on.
                 feature_extractor_name: str,  # name of feature extractor
                 rowcols: List[Tuple[int, int]],
                 classifier_key: str,  # Key to classifier to use.
                 bucketname: str,  # Bucket where classifier is stored.
                 storage_type: str = 's3',
                 ):
        self.pk = pk
        self.im_url = im_url
        self.feature_extractor_name = feature_extractor_name
        self.rowcols = rowcols
        self.classifier_key = classifier_key
        self.bucketname = bucketname
        self.storage_type = storage_type

    @classmethod
    def example(cls):
        return DeployMsg(
            pk=0,
            im_url='www.my.image.jpg',
            feature_extractor_name='vgg16_coralnet_ver1',
            rowcols=[(1, 1), (2, 2)],
            classifier_key='my/classifier/key',
            bucketname='spacer-test'
        )

    @classmethod
    def deserialize(cls, data: Dict) -> 'DeployMsg':
        """ Custom deserializer to convert back to tuples. """
        msg = cls(**data)
        msg.rowcols = [tuple(rc) for rc in data['rowcols']]
        return msg


class DeployReturnMsg(DataClass):
    """ Return message from the deploy task. """

    def __init__(self,
                 model_was_cached: bool,
                 runtime: float,
                 # Scores is a list of scores for every row, col location.
                 scores: List[List[float]],
                 # Maps the score index to a global class id.
                 classes: List[int]):
        self.model_was_cached = model_was_cached
        self.runtime = runtime
        self.scores = scores
        self.classes = classes

    @classmethod
    def example(cls):
        return DeployReturnMsg(
            model_was_cached=True,
            runtime=1.1,
            scores=[[0.1, 0.2, 0.7], [0.9, 0.06, 0.04]],
            classes=[100, 12, 44]
        )


class TaskMsg(DataClass):
    """ Highest level message which hold task messages. """

    def __init__(self,
                 task: str,
                 payload: Union[ExtractFeaturesMsg,
                                TrainClassifierMsg,
                                DeployMsg]):

        assert task in config.TASKS

        self.task = task
        self.payload = payload

    @classmethod
    def deserialize(cls, data: Dict):
        task = data['task']
        payload = data['payload']
        assert task in config.TASKS
        if task == 'extract_features':
            return TaskMsg(task, ExtractFeaturesMsg.deserialize(payload))
        if task == 'train_classifier':
            return TaskMsg(task, TrainClassifierMsg.deserialize(payload))
        if task == 'deploy':
            return TaskMsg(task, DeployMsg.deserialize(payload))

    def serialize(self):
        return {
            'task': self.task,
            'payload': self.payload.serialize()
        }

    @classmethod
    def example(cls):
        return TaskMsg(task='deploy',
                       payload=DeployMsg.example())


class TaskReturnMsg(DataClass):
    """ Highest level return message. """

    def __init__(self,
                 original_job: TaskMsg,
                 ok: bool,
                 results: Optional[Union[ExtractFeaturesReturnMsg,
                                         TrainClassifierReturnMsg,
                                         DeployReturnMsg]],
                 error_message: Optional[str]):

        self.original_job = original_job
        self.results = results
        self.ok = ok
        self.error_message = error_message

    @classmethod
    def example(cls):
        return TaskReturnMsg(
            original_job=TaskMsg.example(),
            ok=True,
            results=DeployReturnMsg.example(),
            error_message=None
        )

    @classmethod
    def deserialize(cls, data: Dict):

        original_job = TaskMsg.deserialize(data['original_job'])

        if data['ok']:
            if original_job.task == 'extract_features':
                results = ExtractFeaturesReturnMsg.deserialize(data['results'])
            elif original_job.task == 'train_classifier':
                results = TrainClassifierReturnMsg.deserialize(data['results'])
            else:
                results = DeployReturnMsg.deserialize(data['results'])
        else:
            results = data['results']

        return TaskReturnMsg(
            original_job=original_job,
            ok=data['ok'],
            results=results,
            error_message=data['error_message']
        )

    def serialize(self):
        return {
            'original_job': self.original_job.serialize(),
            'ok': self.ok,
            'results': self.results.serialize() if self.ok else self.results,
            'error_message': self.error_message
        }
