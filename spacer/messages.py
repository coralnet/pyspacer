"""
Defines message types as data-classes.
Each data-class can serialize itself to a structure of JSON-friendly
python-native data-structures such that it can be stored.
"""

from typing import List, Tuple, Dict, Union, Optional

from spacer import config
from spacer.data_classes import DataClass


class DataLocation(DataClass):
    """
    Points to the location of a piece of data. Can either be a url, a key
    in a s3 bucket, a file path on a local file system or a key to a
    in-memory store.
    """
    def __init__(self,
                 storage_type: str,
                 key: str,
                 bucket_name: Optional[str] = None):

        assert storage_type in config.STORAGE_TYPES, "Storage type not valid."
        if storage_type == 's3':
            assert bucket_name is not None, "Need bucket_name to use s3."
        self.storage_type = storage_type
        self.key = key
        self.bucket_name = bucket_name

    @classmethod
    def example(cls) -> 'DataLocation':
        return DataLocation('memory', 'my_blob')

    @classmethod
    def deserialize(cls, data: Dict) -> 'DataLocation':
        return DataLocation(**data)


class ExtractFeaturesMsg(DataClass):
    """ Input message for extract features class. """

    def __init__(self,
                 job_token: str,  # Token for caller DB reference.
                 feature_extractor_name: str,  # Which extractor to use.
                 rowcols: List[Tuple[int, int]],  # List of [row, col] entries.
                 image_loc: DataLocation,  # Where to fetch image.
                 feature_loc: DataLocation,  # Where to store output.
                 ):

        assert feature_extractor_name in config.FEATURE_EXTRACTOR_NAMES
        assert isinstance(rowcols, List)
        assert len(rowcols) > 0, "Invalid message, rowcols entry is empty."
        assert len(rowcols[0]) == 2
        assert feature_loc.storage_type is not 'url', \
            "Write not supported for url storage type."

        self.job_token = job_token
        self.feature_extractor_name = feature_extractor_name
        self.rowcols = rowcols
        self.image_loc = image_loc
        self.feature_loc = feature_loc

    def serialize(self) -> Dict:
        return {
            'job_token': self.job_token,
            'feature_extractor_name': self.feature_extractor_name,
            'rowcols': list(self.rowcols),
            'image_loc': self.image_loc.serialize(),
            'feature_loc': self.feature_loc.serialize()
        }

    @classmethod
    def deserialize(cls, data: Dict) -> 'ExtractFeaturesMsg':
        """ Custom deserializer required to convert back to tuples. """
        return ExtractFeaturesMsg(
            job_token=data['job_token'],
            feature_extractor_name=data['feature_extractor_name'],
            rowcols=[tuple(rc) for rc in data['rowcols']],
            image_loc=DataLocation.deserialize(data['image_loc']),
            feature_loc=DataLocation.deserialize(data['feature_loc'])
        )

    @classmethod
    def example(cls) -> 'ExtractFeaturesMsg':
        return ExtractFeaturesMsg(
            job_token='123abc',
            feature_extractor_name='vgg16_coralnet_ver1',
            rowcols=[(100, 100)],
            image_loc=DataLocation('memory', 'my_image.jpg'),
            feature_loc=DataLocation('memory', 'my_feats.json'),
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
                 job_token: str,  # Job token, for caller reference.
                 trainer_name: str,  # Name of trainer to use.
                 nbr_epochs: int,  # Number of epochs to do training.
                 traindata_loc: DataLocation,  # Traindata
                 valdata_loc: DataLocation,  # Valdata
                 features_loc: DataLocation,  # Location of features. Key is set from traindata and valdata during training.
                 previous_model_locs: List[DataLocation],  # Previous models to be evaluated on the valdata.
                 model_loc: DataLocation,  # Where to store model.
                 valresult_loc: DataLocation,  # Model result on val.
                 ):

        assert trainer_name in config.TRAINER_NAMES

        self.job_token = job_token
        self.trainer_name = trainer_name
        self.nbr_epochs = nbr_epochs
        self.traindata_loc = traindata_loc
        self.valdata_loc = valdata_loc
        self.features_loc = features_loc
        self.previous_model_locs = previous_model_locs
        self.model_loc = model_loc
        self.valresult_loc = valresult_loc

    @classmethod
    def example(cls):
        return TrainClassifierMsg(
            job_token='123_abc',
            trainer_name='minibatch',
            nbr_epochs=2,
            traindata_loc=DataLocation('memory', 'my_traindata.json'),
            valdata_loc=DataLocation('memory', 'my_valdata.json'),
            features_loc=DataLocation('memory', ''),
            previous_model_locs=[
                DataLocation('memory', 'previous_model1.pkl'),
                DataLocation('memory', 'previous_model2.pkl'),
            ],
            model_loc=DataLocation('memory', 'my_new_model.pkl'),
            valresult_loc=DataLocation('memory', 'my_valresult.json')
        )

    def serialize(self) -> Dict:
        return {
            'job_token': self.job_token,
            'trainer_name': self.trainer_name,
            'nbr_epochs': self.nbr_epochs,
            'traindata_loc': self.traindata_loc.serialize(),
            'valdata_loc': self.valdata_loc.serialize(),
            'features_loc': self.features_loc.serialize(),
            'previous_model_locs': [entry.serialize()
                                    for entry in self.previous_model_locs],
            'model_loc': self.model_loc.serialize(),
            'valresult_loc': self.valresult_loc.serialize(),
        }

    @classmethod
    def deserialize(cls, data: Dict) -> 'TrainClassifierMsg':
        """ Redefining to help pycharm typing module """
        return TrainClassifierMsg(
            job_token=data['job_token'],
            trainer_name=data['trainer_name'],
            nbr_epochs=data['nbr_epochs'],
            traindata_loc=DataLocation.deserialize(data['traindata_loc']),
            valdata_loc=DataLocation.deserialize(data['valdata_loc']),
            features_loc=DataLocation.deserialize(data['features_loc']),
            previous_model_locs=[DataLocation.deserialize(entry)
                                 for entry in data['previous_model_locs']],
            model_loc=DataLocation.deserialize(data['model_loc']),
            valresult_loc=DataLocation.deserialize(data['valresult_loc'])
        )


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
                                     key='https://spacer-test.s3-us-west-2.'
                                         'amazonaws.com/08bfc10v7t.png.'
                                         'featurevector'),
            classifier_loc=DataLocation(storage_type='url',
                                        key='https://spacer-test.s3-us-west-2.'
                                        'amazonaws.com/legacy.model')
        )

    def serialize(self):
        return {
            'job_token': self.job_token,
            'feature_loc': self.feature_loc.serialize(),
            'classifier_loc': self.classifier_loc.serialize(),
        }

    @classmethod
    def deserialize(cls, data: Dict):
        return ClassifyFeaturesMsg(
            job_token=data['job_token'],
            feature_loc=DataLocation.deserialize(data['feature_loc']),
            classifier_loc=DataLocation.deserialize(data['classifier_loc'])
        )


class ClassifyImageMsg(DataClass):
    """ Specifies the classify_image task. """

    def __init__(self,
                 job_token: str,  # Primary key of job, not used in spacer.
                 image_loc: DataLocation,  # Location of image to classify.
                 feature_extractor_name: str,  # name of feature extractor
                 rowcols: List[Tuple[int, int]],
                 classifier_loc: DataLocation,  # Location of classifier to use.
                 ):
        self.job_token = job_token
        self.image_loc = image_loc
        self.feature_extractor_name = feature_extractor_name
        self.rowcols = rowcols
        self.classifier_loc = classifier_loc

    @classmethod
    def example(cls):
        return ClassifyImageMsg(
            job_token='my_job',
            image_loc=DataLocation(storage_type='url',
                                   key='https://spacer-test.s3-us-west-2.'
                                   'amazonaws.com/08bfc10v7t.png'),
            feature_extractor_name='vgg16_coralnet_ver1',
            rowcols=[(1, 1), (2, 2)],
            classifier_loc=DataLocation(storage_type='url',
                                        key='https://spacer-test.s3-us-west-2.'
                                        'amazonaws.com/legacy.model')
        )

    def serialize(self):
        return {
            'job_token': self.job_token,
            'image_loc': self.image_loc.serialize(),
            'feature_extractor_name': self.feature_extractor_name,
            'rowcols': self.rowcols,
            'classifier_loc': self.classifier_loc.serialize()
        }

    @classmethod
    def deserialize(cls, data: Dict) -> 'ClassifyImageMsg':
        """ Custom deserializer to convert back to tuples. """
        return ClassifyImageMsg(
            job_token=data['job_token'],
            image_loc=DataLocation.deserialize(data['image_loc']),
            feature_extractor_name=data['feature_extractor_name'],
            rowcols=[(row, col) for row, col in data['rowcols']],
            classifier_loc=DataLocation.deserialize(data['classifier_loc'])
        )


class ClassifyReturnMsg(DataClass):
    """ Return message from the classify_{image, features} tasks. """

    def __init__(self,
                 runtime: float,
                 # Scores is a list of (row, col, [scores]) tuples.
                 scores: List[Tuple[int, int, List[float]]],
                 # Maps the score index to a global class id.
                 classes: List[int]):
        self.runtime = runtime
        self.scores = scores
        self.classes = classes

    @classmethod
    def example(cls):
        return ClassifyReturnMsg(
            runtime=1.1,
            scores=[(10, 20, [0.1, 0.2, 0.7]), (20, 40, [0.9, 0.06, 0.04])],
            classes=[100, 12, 44]
        )

    @classmethod
    def deserialize(cls, data: Dict) -> 'ClassifyReturnMsg':
        return ClassifyReturnMsg(
            runtime=data['runtime'],
            scores=[(row, col, scores) for
                    row, col, scores in data['scores']],
            classes=data['classes']
        )


class TaskMsg(DataClass):
    """ Highest level message which hold task messages. """

    def __init__(self,
                 task: str,
                 payload: Union[ExtractFeaturesMsg,
                                TrainClassifierMsg,
                                ClassifyFeaturesMsg,
                                ClassifyImageMsg]):

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
        if task == 'classify_features':
            return TaskMsg(task, ClassifyFeaturesMsg.deserialize(payload))
        if task == 'classify_image':
            return TaskMsg(task, ClassifyImageMsg.deserialize(payload))

    def serialize(self):
        return {
            'task': self.task,
            'payload': self.payload.serialize()
        }

    @classmethod
    def example(cls):
        return TaskMsg(task='classify_image',
                       payload=ClassifyImageMsg.example())


class TaskReturnMsg(DataClass):
    """ Highest level return message. """

    def __init__(self,
                 original_job: TaskMsg,
                 ok: bool,
                 results: Optional[Union[ExtractFeaturesReturnMsg,
                                         TrainClassifierReturnMsg,
                                         ClassifyReturnMsg]],
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
            results=ClassifyReturnMsg.example(),
            error_message=None
        )

    @classmethod
    def deserialize(cls, data: Dict):

        original_job = TaskMsg.deserialize(data['original_job'])

        if data['ok']:
            task_name = original_job.task
            assert task_name in config.TASKS
            if task_name == 'extract_features':
                results = ExtractFeaturesReturnMsg.deserialize(data['results'])
            elif task_name == 'train_classifier':
                results = TrainClassifierReturnMsg.deserialize(data['results'])
            else:  # task_name in ['classify_image', 'classify_features']
                results = ClassifyReturnMsg.deserialize(data['results'])
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
