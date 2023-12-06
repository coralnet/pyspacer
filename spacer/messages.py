"""
Defines message types as data-classes.
Each data-class can serialize itself to a structure of JSON-friendly
python-native data-structures such that it can be stored.
"""

from __future__ import annotations
from pathlib import Path
from urllib.parse import urlparse

from spacer import config
from spacer.data_classes import DataClass, ImageLabels


class DataLocation(DataClass):
    """
    Points to the location of a piece of data. Can either be a url, a key
    in a s3 bucket, a file path on a local file system or a key to a
    in-memory store.
    """
    def __init__(self,
                 storage_type: str,
                 key: str,
                 bucket_name: str | None = None):

        assert storage_type in config.STORAGE_TYPES, "Storage type not valid."
        if storage_type == 's3':
            assert bucket_name is not None, "Need bucket_name to use s3."
        self.storage_type = storage_type
        self.key = key
        self.bucket_name = bucket_name

    @classmethod
    def example(cls) -> 'DataLocation':
        return DataLocation('memory', 'my_blob')

    @property
    def filename(self) -> str:
        if self.storage_type == 'url':
            # This is a basic implementation which just gets the last
            # part of the URL 'path', even if that part isn't filename-like.
            return Path(urlparse(self.key).path).name
        else:
            return Path(self.key).name

    @property
    def is_remote(self) -> bool:
        return self.storage_type in ['url', 's3']

    @property
    def is_writable(self) -> bool:
        return self.storage_type != 'url'

    @classmethod
    def deserialize(cls, data: dict) -> 'DataLocation':
        return DataLocation(**data)

    def __hash__(self):
        return hash((self.storage_type, self.key, self.bucket_name))


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
        from spacer.extract_features import EfficientNetExtractor
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
        from spacer.extract_features import FeatureExtractor
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


class TrainClassifierMsg(DataClass):
    """ Specifies the train classifier task. """

    def __init__(
        self,
        # For caller's reference.
        job_token: str,
        # 'minibatch' is currently the only trainer that spacer defines.
        trainer_name: str,
        # How many iterations the training algorithm should run; more epochs
        # = more opportunity to converge to a better fit, but slower.
        nbr_epochs: int,
        # Classifier types available:
        # 1. 'MLP': multi-layer perceptron; newer classifier type for CoralNet
        # 2. 'LR': logistic regression; older classifier type for CoralNet
        clf_type: str,
        # Point-locations to ground-truth-labels (annotations) mapping.
        # Used for training the classifier. Two acceptable formats:
        #
        # 1) One ImageLabels instance, which pyspacer decides how to
        # split up into training, reference, and validation sets.
        # 2) A dict containing three ImageLabels instances, with the
        # dict keys being 'train', 'ref', and 'val'.
        #
        # For option 2, the reference set's label count should be no greater
        # than config.TRAINING_BATCH_LABEL_COUNT.
        #
        # The reference set is a hold-out set whose purpose is to
        # 1) get an accuracy measurement per epoch, and
        # 2) calibrate classifier output scores.
        # The validation set is used to evaluate accuracy of the final
        # classifier.
        labels: ImageLabels | dict[str, ImageLabels],
        # All the feature vectors should use the same storage_type, and the
        # same S3 bucket_name if applicable. This DataLocation's purpose is
        # to describe those common storage details. The key arg is ignored,
        # because that will be different for each feature vector.
        features_loc: DataLocation,
        # List of previously-created models (classifiers) to also evaluate
        # using this dataset, for informational purposes only.
        # A classifier is stored as a pickled CalibratedClassifierCV.
        previous_model_locs: list[DataLocation],
        # Where the new model (classifier) should be output to.
        model_loc: DataLocation,
        # Where the detailed evaluation results of the new model should be
        # stored.
        valresult_loc: DataLocation,
    ):

        assert trainer_name in config.TRAINER_NAMES

        self.job_token = job_token
        self.trainer_name = trainer_name
        self.nbr_epochs = nbr_epochs
        self.clf_type = clf_type
        self.features_loc = features_loc
        self.previous_model_locs = previous_model_locs
        self.model_loc = model_loc
        self.valresult_loc = valresult_loc

        if isinstance(labels, dict):
            # The caller has decided how to split the data into
            # training, reference, and validation sets.
            self.labels = labels
        else:
            # Split data into training, reference, and validation sets.
            # Arbitrarily, validation gets 10%, reference gets
            # min(10%, TRAINING_BATCH_LABEL_COUNT), training gets the rest.
            # This is imprecise because it's split on the image level, not the
            # label level, and images can have different numbers of labels.
            train_data = dict()
            ref_data = dict()
            val_data = dict()
            ref_label_count = 0
            ref_done = False

            for image_index, image_key in enumerate(labels.image_keys):
                this_image_labels = labels.data[image_key]

                if image_index % 10 == 0:
                    # 1st, 11th, 21st, etc. images go in val.
                    val_data[image_key] = this_image_labels
                elif not ref_done and image_index % 10 == 1:
                    # 2nd, 12th, 22nd, etc. images go in ref, if ref still
                    # has room within the batch size; else, go in train.
                    if (ref_label_count + len(this_image_labels)
                            <= config.TRAINING_BATCH_LABEL_COUNT):
                        ref_data[image_key] = this_image_labels
                        ref_label_count += len(this_image_labels)
                    else:
                        # ref would go over the batch size if it added this
                        # image.
                        train_data[image_key] = this_image_labels
                        ref_done = True
                else:
                    # The rest go in train.
                    train_data[image_key] = this_image_labels

            self.labels = dict(
                train=ImageLabels(train_data),
                ref=ImageLabels(ref_data),
                val=ImageLabels(val_data),
            )

    @classmethod
    def example(cls):
        return TrainClassifierMsg(
            job_token='123_abc',
            trainer_name='minibatch',
            nbr_epochs=2,
            clf_type='MLP',
            labels=ImageLabels.example(),
            features_loc=DataLocation('memory', ''),
            previous_model_locs=[
                DataLocation('memory', 'previous_model1.pkl'),
                DataLocation('memory', 'previous_model2.pkl'),
            ],
            model_loc=DataLocation('memory', 'my_new_model.pkl'),
            valresult_loc=DataLocation('memory', 'my_valresult.json')
        )

    def serialize(self) -> dict:
        return {
            'job_token': self.job_token,
            'trainer_name': self.trainer_name,
            'nbr_epochs': self.nbr_epochs,
            'clf_type': self.clf_type,
            'labels': dict([
                (key, entry.serialize())
                for key, entry in self.labels.items()
            ]),
            'features_loc': self.features_loc.serialize(),
            'previous_model_locs': [entry.serialize()
                                    for entry in self.previous_model_locs],
            'model_loc': self.model_loc.serialize(),
            'valresult_loc': self.valresult_loc.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> 'TrainClassifierMsg':
        return TrainClassifierMsg(
            job_token=data['job_token'],
            trainer_name=data['trainer_name'],
            nbr_epochs=data['nbr_epochs'],
            clf_type=data['clf_type'],
            labels=dict([
                (key, ImageLabels.deserialize(entry))
                for key, entry in data['labels'].items()
            ]),
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
        from spacer.extract_features import EfficientNetExtractor
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
        from spacer.extract_features import FeatureExtractor
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
                 classes: list[int],
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
