"""
Defines data-classes for input and output types.
Each data-class can serialize itself to a structure of JSON-friendly
python-native data-structures such that it can be stored.
"""
import json
import logging
from abc import ABC, abstractmethod
from io import BytesIO
from pprint import pformat
from typing import Dict, List, Tuple, Set, Optional, Union

import numpy as np

from spacer.storage import storage_factory


class DataClass(ABC):  # pragma: no cover

    @classmethod
    def load(cls, loc: 'DataLocation'):
        storage = storage_factory(loc.storage_type, loc.bucket_name)
        return cls.deserialize(json.loads(
            storage.load(loc.key).getvalue().decode('utf-8')))

    def store(self, loc: 'DataLocation'):
        logging.info('Storing to {}'.format(loc.serialize()))
        storage = storage_factory(loc.storage_type, loc.bucket_name)
        storage.store(loc.key, BytesIO(
            json.dumps(self.serialize()).encode('utf-8')))
        logging.info('Done storing to {}'.format(loc.serialize()))

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
        """ Give a (sort of) nicely formatted string repr. of class. """
        return pformat(vars(self))

    def __eq__(self, other):
        """
        This only works for the simple classes, more complicated
        need to overwrite this method
        """
        sd = self.__dict__
        od = other.__dict__
        return sd.keys() == od.keys() and all([sd[key] == od[key]
                                               for key in sd])


class ImageLabels(DataClass):
    """ Contains row, col, label information for an image. """

    def __init__(self,
                 # Data maps a feature key (or file path) to a List of
                 # (row, col, label).
                 data: Dict[str, List[Tuple[int, int, int]]]):
        self.data = data

    @classmethod
    def example(cls):
        return ImageLabels({
            'img1.features': [(100, 200, 3), (101, 200, 2)],
            'img2.features': [(100, 202, 3), (101, 200, 3)],
            'img3.features': [(100, 202, 3), (101, 200, 3)],
            'img4.features': [(100, 202, 3), (101, 200, 3)],
        })

    @classmethod
    def deserialize(cls, data: Dict) -> 'ImageLabels':
        """ Custom deserializer required to convert back to tuples. """
        return ImageLabels(
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
            labels |= {label for (row, col, label) in self.data[im_key]}
        return labels

    def __len__(self):
        return len(self.data)


class PointFeatures(DataClass):
    """ Contains row, col, feature-vector for a single point. """

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

    @property
    def data_np(self):
        return np.array(self.data).reshape(1, -1)


class ImageFeatures(DataClass):
    """ Contains row, col, feature-vectors for an image. """

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

    def get_array(self, rowcol: Tuple[int, int]) -> np.array:
        """
        Similar to __getitem__ but returns a numpy array formatted
        correctly for classification by a classifier of type
        sklearn.calibration.CalibratedClassifierCV
        """
        return np.array(self[rowcol]).reshape(1, -1)

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

    @classmethod
    def deserialize(cls, data: Union[Dict, List]) -> 'ImageFeatures':

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

    def serialize(self):
        return {
            'point_features': [feats.serialize() for
                               feats in self.point_features],
            'valid_rowcol': self.valid_rowcol,
            'feature_dim': self.feature_dim,
            'npoints': self.npoints
        }

    def __eq__(self, other):
        return all([a == b for a, b in zip(self.point_features,
                                           other.point_features)]) and \
               self.valid_rowcol == other.valid_rowcol and \
               self.feature_dim == other.feature_dim and \
               self.npoints == other.npoints

    @classmethod
    def make_random(cls,
                    point_labels: List[int],
                    feature_dim: int):
        pfs = [PointFeatures(row=itt,
                             col=itt,
                             data=list(label + np.random.randn(feature_dim)))
               for itt, label in enumerate(point_labels)]

        return ImageFeatures(point_features=pfs,
                             valid_rowcol=True,
                             feature_dim=feature_dim,
                             npoints=len(point_labels))


class ValResults(DataClass):
    """ Defines the validation results data class. Note that the gt and est
    lists points to the index into the classes list."""

    def __init__(self,
                 scores: List[float],
                 gt: List[int],  # Using singular for backwards compatibility.
                 est: List[int],  # Using singular for backwards compatibility.
                 classes: List[int]):

        assert len(gt) == len(est)
        assert len(gt) == len(scores)
        assert max(gt) < len(classes)
        assert max(est) < len(classes)
        assert min(gt) >= 0
        assert min(est) >= 0

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

    @classmethod
    def deserialize(cls, data: Dict) -> 'ValResults':
        """ Redefined here to help the Typing module. """
        return ValResults(**data)
