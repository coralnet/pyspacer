from typing import List, Tuple, Dict, Union, Optional
from pprint import pformat
from abc import ABC, abstractmethod

from spacer import config


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
        Serialized the class content to native data-types, dicts, lists, etc, such that
        it it compatible with json.dumps and json.loads.
        """
        return self.__dict__

    def __repr__(self):
        return pformat(vars(self))

    def __eq__(self, other):
        sd = self.__dict__
        od = other.__dict__
        return sd.keys() == od.keys() and all([sd[key] == od[key] for key in sd])


class ExtractFeaturesMsg(DataClass):

    def __init__(self,
                 pk: int,
                 modelname: str,
                 bucketname: str,
                 imkey: str,
                 rowcols: List[Tuple[int, int]],
                 outputkey: str,
                 storage_type: str = 's3'):

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
        msg.rowcols = [tuple(entry) for entry in msg.rowcols]  # JSONstores tuples as lists, we restore it here.
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
                 runtime: Dict[str, float]):

        self.model_was_cashed = model_was_cashed
        self.runtime = runtime

    @classmethod
    def example(cls) -> 'ExtractFeaturesReturnMsg':
        return ExtractFeaturesReturnMsg(
            model_was_cashed=True,
            runtime={
                'total': 2.0,
                'core': 1.0,
                'per_point': 0.1
            }
        )


class TrainClassifierMsg(DataClass):

    def __init__(self,
                 pk: int,  # Primary key of the model to train
                 bucketname: str,  # Bucket name where features are stored.
                 traindata: str,  # Struct defining labels and features files.
                 model: str,  # Key for where to store the trained model.
                 valdata: str,  # Structure defining previous models and performances.
                 ):

        self.pk = pk
        self.bucketname = bucketname
        self.traindata = traindata
        self.model = model
        self.valdata = valdata

    @classmethod
    def example(cls):
        return TrainClassifierMsg(
            pk=1,
            bucketname='spacer-test',
            traindata='s3bucketkey',
            model='my_trained_model',
            valdata='my_valdata'
        )


class TrainClassifierReturnMsg(DataClass):

    def __init__(self,
                 acc: float,  # Accuracy of new classifier on the validation set.
                 pc_accs: List[float],  # Accuracy of previous classifiers on the validation set.
                 refacc: List[float],  # Accuracy on reference set for each epoch of training.
                 runtime: float,  # Runtime for full training execution.
                 ):
        self.acc = acc
        self.pc_accs = pc_accs
        self.refacc = refacc
        self.runtime = runtime

    @classmethod
    def example(cls):
        return TrainClassifierReturnMsg(
            acc=0.7,
            pc_accs=[0.4, 0.5, 0.6],
            refacc=[0.55, 0.65, 0.64, 0.67, 0.70],
            runtime=123.4,
        )


class PointFeatures(DataClass):

    def __init__(self,
                 row: int,  # Row which this feature vector was extracted
                 col: int,  # Colum for which this feature vector was extracted
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
                 point_features: List[PointFeatures],  # List of features for all points in image
                 valid_rowcol: bool,  # Legacy feature did not store row and column locations.
                 feature_dim: int,  # Dimensionality of the feature vectors.
                 npoints: int,  # Number of points in this image (same as len(self.point_features)).
                 ):

        assert len(point_features) == npoints
        assert len(point_features) > 0
        assert len(point_features[0].data) == feature_dim

        self.point_features = point_features
        self.valid_rowcol = valid_rowcol
        self.feature_dim = feature_dim
        self.npoints = npoints

    @classmethod
    def example(cls):
        point_feature = PointFeatures.example()
        return cls(
            point_features=[point_feature, point_feature],
            valid_rowcol=True,
            feature_dim=len(point_feature.data),
            npoints=2
        )

    def serialize(self):
        return {
            'point_features': [feats.serialize() for feats in self.point_features],
            'valid_rowcol': self.valid_rowcol,
            'feature_dim': self.feature_dim,
            'npoints': self.npoints
        }

    @classmethod
    def deserialize(cls, data: Union[Dict, List]):

        if isinstance(data, List):
            # Legacy feature were stored as a list of list without row and column information.
            assert len(data) > 0, "Empty features file."
            return ImageFeatures(
                point_features=[PointFeatures(row=0, col=0, data=entry) for entry in data],
                valid_rowcol=False,
                feature_dim=len(data[0]),
                npoints=len(data)
            )
        else:
            return ImageFeatures(
                point_features=[PointFeatures.deserialize(feat) for feat in data['point_features']],
                valid_rowcol=data['valid_rowcol'],
                feature_dim=data['feature_dim'],
                npoints=data['npoints']
            )

    def __eq__(self, other):
        return all([a == b for a, b in zip(self.point_features, other.point_features)]) and \
               self.valid_rowcol == other.valid_rowcol and \
               self.feature_dim == other.feature_dim and \
               self.npoints == other.npoints


class DeployMsg:
    pass


class DeployReturnMsg:
    pass


class TaskMsg:

    def __init__(self,
                 task: str,
                 payload: Union[ExtractFeaturesMsg, TrainClassifierMsg, DeployMsg]):

        self.task = task
        self.payload = payload


class TaskReturnMsg:

    def __init__(self,
                 original_job: TaskMsg,
                 ok: bool,
                 results: Optional[Union[ExtractFeaturesMsg, TrainClassifierReturnMsg, DeployReturnMsg]],
                 error_message: Optional[str]):

        self.original_job = original_job
        self.results = results
        self.ok = ok
        self.error_message = error_message