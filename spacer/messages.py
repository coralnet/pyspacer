import json

from abc import ABC, abstractmethod
from pprint import pformat
from typing import List, Tuple, Dict, Union, Optional, Set

from spacer import config
from spacer.storage import Storage


class DataClass(ABC):

    @classmethod
    @abstractmethod
    def example(cls):
        """
        Instantiate an example member of the class.
        Useful for testing, tutorials, etc.
        """

    @classmethod
    def deserialize(cls, data: Dict) -> 'DataClass':
        """
        Initializes a class instance from input data.
        Input should be a dictionary containing native python data structures.
        """
        return cls(**data)

    def serialize(self) -> Dict:
        """
        Serialized the class content to native data-types, dicts, lists, etc,
        such that it it compatible with json.dumps and json.loads.
        """
        return self.__dict__

    def __repr__(self):
        return pformat(vars(self))

    def __eq__(self, other):
        sd = self.__dict__
        od = other.__dict__
        return sd.keys() == od.keys() and \
               all([sd[key] == od[key] for key in sd])


class ExtractFeaturesMsg(DataClass):

    def __init__(self,
                 pk: int,
                 modelname: str,
                 bucketname: str,
                 imkey: str,
                 rowcols: List[Tuple[int, int]],  # List of [row, col] entries.
                 outputkey: str,
                 storage_type: str = 's3',
                 ):

        assert storage_type in config.STORAGE_TYPES
        assert modelname in config.FEATURE_EXTRACTOR_NAMES
        assert isinstance(rowcols, List)
        assert len(rowcols) > 0, "Invalid message, rowcols entry is empty."
        assert len(rowcols[0]) == 2

        self.pk = pk
        self.modelname = modelname
        self.bucketname = bucketname
        self.imkey = imkey
        self.storage_type = storage_type
        self.rowcols = rowcols
        self.outputkey = outputkey

    @classmethod
    def deserialize(cls, data: Dict) -> 'ExtractFeaturesMsg':
        msg = cls(**data)
        """ Custom deserializer to convert back to tuples. """
        msg.rowcols = [tuple(rc) for rc in data['rowcols']]
        return msg

    @classmethod
    def example(cls) -> 'ExtractFeaturesMsg':
        return ExtractFeaturesMsg(
            pk=1,
            modelname='vgg16_coralnet_ver1',
            bucketname='spacer-test',
            imkey='edinburgh3.jpg',
            rowcols=[(100, 100)],
            outputkey='edinburgh3.jpg.feats',
            storage_type='s3',
        )


class ExtractFeaturesReturnMsg(DataClass):

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


class FeatureLabels(DataClass):

    def __init__(self,
                 # Data maps a feature key (or file path) to a List of
                 # [row, col, label].
                 data: Dict[str, List[Tuple[int, int, int]]]):
        self.data = data

    @classmethod
    def example(cls):
        return FeatureLabels({
            'img1.features': [(100, 200, 3), (101, 200, 2)],
            'img2.features': [(100, 202, 3), (101, 200, 3)],
            'img3.features': [(100, 202, 3), (101, 200, 3)],
            'img4.features': [(100, 202, 3), (101, 200, 3)],
        })

    @classmethod
    def deserialize(cls, data: Dict) -> 'FeatureLabels':
        """ Custom deserializer to convert back to tuples. """
        return FeatureLabels(
            data={key: [tuple(entry) for entry in value] for
                  key, value in data['data'].items()})

    @property
    def image_keys(self):
        return list(self.data.keys())

    @property
    def samples_per_image(self):
        return len(next(iter(self.data.values())))

    def unique_classes(self, key_list: List[str]) -> Set[int]:
        """ Returns the set of all unique classes in the key_list subset. """
        labels = set()
        for im_key in key_list:
            labels += set([label for (row, col, label) in self.data[im_key]])
        return labels

    def __len__(self):
        return len(self.data)


class TrainClassifierMsg(DataClass):

    def __init__(self,
                 # Primary key of the model to train
                 pk: int,
                 # Key for where to store the trained model.
                 model_key: str,
                 # Key to LabeledFeatures structure with training data.
                 traindata_key: str,
                 # Key to LabeledFeatures structure with validation data.
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

        self.pk = pk
        self.model_key = model_key
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

    def load_train_feature_labels(self, storage: Storage) -> FeatureLabels:
        return FeatureLabels.deserialize(
            json.loads(storage.load_string(self.traindata_key)))

    def load_val_feature_labels(self, storage: Storage) -> FeatureLabels:
        return FeatureLabels.deserialize(
            json.loads(storage.load_string(self.valdata_key)))


class TrainClassifierReturnMsg(DataClass):

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


class DeployMsg:
    pass


class DeployReturnMsg:
    pass


class TaskMsg:

    def __init__(self,
                 task: str,
                 payload: Union[ExtractFeaturesMsg,
                                TrainClassifierMsg,
                                DeployMsg]):

        self.task = task
        self.payload = payload


class TaskReturnMsg:

    def __init__(self,
                 original_job: TaskMsg,
                 ok: bool,
                 results: Optional[Union[ExtractFeaturesMsg,
                                         TrainClassifierReturnMsg,
                                         DeployReturnMsg]],
                 error_message: Optional[str]):

        self.original_job = original_job
        self.results = results
        self.ok = ok
        self.error_message = error_message


class PointFeatures(DataClass):

    def __init__(self,
                 row: Optional[int],  # Row where feature was extracted
                 col: Optional[int],  # Column where feature was extracted
                 data: List[float],  # Feature vector as list of floats.
                 ):
        self.row = row
        self.col = col
        self.data = data

    @classmethod
    def example(cls):
        return cls(
            row=100,
            col=100,
            data=[1.1, 1.3, 1.12]
        )

    @classmethod
    def deserialize(cls, data: Dict) -> 'PointFeatures':
        """ Redefined here to help the Typing module. """
        return cls(**data)


class ImageFeatures(DataClass):

    def __init__(self,
                 # List of features for all points in image
                 point_features: List[PointFeatures],
                 # Legacy feature did not store row and column locations.
                 valid_rowcol: bool,
                 # Dimensionality of the feature vectors.
                 feature_dim: int,
                 # Number of points in this image.
                 npoints: int,
                 ):

        assert len(point_features) == npoints
        assert len(point_features) > 0
        assert len(point_features[0].data) == feature_dim

        self.point_features = point_features
        self.valid_rowcol = valid_rowcol
        self.feature_dim = feature_dim
        self.npoints = npoints

        if self.valid_rowcol:
            # Store a row_col hash for quick retrieval based on (row, col)
            self._rchash = {(pf.row, pf.col): enum for
                            enum, pf in enumerate(self.point_features)}

    def __getitem__(self, rowcol: Tuple[int, int]) -> List[float]:
        """ Returns features at (row, col) location. """
        if not self.valid_rowcol:
            raise ValueError('Method not supported for legacy features')
        return self.point_features[self._rchash[rowcol]].data

    @classmethod
    def example(cls):
        pf1 = PointFeatures(row=100, col=100, data=[1.1, 1.3, 1.12])
        pf2 = PointFeatures(row=120, col=110, data=[1.9, 1.3, 1.12])
        return cls(
            point_features=[pf1, pf2],
            valid_rowcol=True,
            feature_dim=len(pf1.data),
            npoints=2
        )

    def serialize(self):
        return {
            'point_features': [feats.serialize() for
                               feats in self.point_features],
            'valid_rowcol': self.valid_rowcol,
            'feature_dim': self.feature_dim,
            'npoints': self.npoints
        }

    @classmethod
    def deserialize(cls, data: Union[Dict, List]):

        if isinstance(data, List):
            # Legacy feature were stored as a list of list
            # without row and column information.
            assert len(data) > 0, "Empty features file."
            return ImageFeatures(
                point_features=[PointFeatures(row=None,
                                              col=None,
                                              data=entry) for entry in data],
                valid_rowcol=False,
                feature_dim=len(data[0]),
                npoints=len(data)
            )
        else:
            return ImageFeatures(
                point_features=[PointFeatures.deserialize(feat)
                                for feat in data['point_features']],
                valid_rowcol=data['valid_rowcol'],
                feature_dim=data['feature_dim'],
                npoints=data['npoints']
            )

    def __eq__(self, other):
        return all([a == b for a, b in zip(self.point_features,
                                           other.point_features)]) and \
               self.valid_rowcol == other.valid_rowcol and \
               self.feature_dim == other.feature_dim and \
               self.npoints == other.npoints


class ValResults(DataClass):

    def __init__(self,
                 scores: List[float],
                 gt: List[int],  # Using singular for backwards compatibility.
                 est: List[int],  # Using singular for backwards compatibility.
                 classes: List[int]):

        self.scores = scores
        self.gt = gt
        self.est = est
        self.classes = classes

    @classmethod
    def example(cls):
        return cls(scores=[.9, .8, .7],
                   gt=[0, 1, 0],
                   est=[0, 1, 1],
                   classes=[121, 1222])

