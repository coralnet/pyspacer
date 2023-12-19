"""
Defines data-classes for input and output types.
Each data-class can serialize itself to a structure of JSON-friendly
python-native data-structures such that it can be stored.
"""
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from io import BytesIO
from pprint import pformat

import numpy as np

from spacer.storage import storage_factory


class DataClass(ABC):  # pragma: no cover

    @classmethod
    def load(cls, loc: 'DataLocation'):
        storage = storage_factory(loc.storage_type, loc.bucket_name)
        return cls.deserialize(json.loads(
            storage.load(loc.key).getvalue().decode('utf-8')))

    def store(self, loc: 'DataLocation'):
        storage = storage_factory(loc.storage_type, loc.bucket_name)
        storage.store(loc.key, BytesIO(
            json.dumps(self.serialize()).encode('utf-8')))

    @classmethod
    @abstractmethod
    def example(cls):
        """
        Instantiate an example member of the class.
        Useful for testing, tutorials, etc.
        """

    @classmethod
    def deserialize(cls, data: dict) -> 'DataClass':
        """
        Initializes a class instance from input data.
        Input should be a dictionary containing native python data structures.
        """
        return cls(**data)

    def serialize(self) -> dict:
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
    """ Contains row, col, label information for a set of images. """

    def __init__(self,
                 # Data maps a feature key (or file path) to a List of
                 # (row, col, label).
                 data: dict[str, list[tuple[int, int, int]]]):
        self._data = data

        self.label_count = sum([len(labels) for labels in data.values()])

        self.classes_set = set()
        for single_image_labels in data.values():
            single_image_classes = set(
                label for (row, col, label) in single_image_labels)
            self.classes_set |= single_image_classes

    @classmethod
    def example(cls):
        return ImageLabels({
            'img1.features': [(100, 200, 3), (101, 200, 2)],
            'img2.features': [(100, 202, 3), (101, 200, 3)],
            'img3.features': [(100, 202, 3), (101, 200, 3)],
            'img4.features': [(100, 202, 3), (101, 200, 3)],
        })

    def serialize(self) -> dict:
        """Only need the `_data` field; the other fields can be recomputed."""
        return self._data

    @classmethod
    def deserialize(cls, data: dict) -> 'ImageLabels':
        """Custom deserializer required to convert back to tuples."""
        return ImageLabels({
            key: [tuple(entry) for entry in value] for
            key, value in data.items()
        })

    @property
    def image_keys(self):
        return list(self._data.keys())

    def filter_classes(self, accepted_classes) -> 'ImageLabels':
        """
        Make a new instance by filtering out labels not included in
        the specified classes.
        """
        data = {}
        for image_key in self.image_keys:
            this_image_labels = [
                (row, column, label)
                for row, column, label in self._data[image_key]
                if label in accepted_classes
            ]
            # Only include an image if it has any labels remaining
            # after filtering.
            if len(this_image_labels) > 0:
                data[image_key] = this_image_labels
        return ImageLabels(data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __contains__(self, item):
        return item in self._data


class PointFeatures(DataClass):
    """ Contains row, col, feature-vector for a single point. """

    def __init__(self,
                 row: int | None,  # Row where feature was extracted
                 col: int | None,  # Column where feature was extracted
                 data: np.array,  # Feature vector with 32 bit precision.
                 ):
        self.row = row
        self.col = col
        self.data = np.array(data, dtype=float)

    @classmethod
    def example(cls):
        return cls(
            row=100,
            col=100,
            data=np.array([1.1, 1.3, 1.12], dtype=float)
        )

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col and \
               np.allclose(self.data, other.data)


class ImageFeatures(DataClass):
    """ Contains row, col, feature-vectors for an image. """

    def __init__(self,
                 # List of features for all points in image
                 point_features: list[PointFeatures],
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

    def __getitem__(self, rowcol: tuple[int, int]) -> list[float]:
        """ Returns features at (row, col) location. """
        if not self.valid_rowcol:
            raise ValueError('Method not supported for legacy features')
        return self.point_features[self._rchash[rowcol]].data

    def get_array(self, rowcol: tuple[int, int]) -> np.array:
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
    def deserialize(cls, data: dict | list) -> 'ImageFeatures':

        assert isinstance(data, list), \
            "Deserialize only supported for legacy format"
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

    def serialize(self):
        raise NotImplementedError('Use .store() and .load() methods instead.')

    def __eq__(self, other):
        return all([a == b for a, b in zip(self.point_features,
                                           other.point_features)]) and \
               self.valid_rowcol == other.valid_rowcol and \
               self.feature_dim == other.feature_dim and \
               self.npoints == other.npoints

    @classmethod
    def make_random(cls,
                    point_labels: list[int],
                    feature_dim: int):
        pfs = [PointFeatures(row=itt,
                             col=itt,
                             data=list(label + np.random.randn(feature_dim)))
               for itt, label in enumerate(point_labels)]

        return ImageFeatures(point_features=pfs,
                             valid_rowcol=True,
                             feature_dim=feature_dim,
                             npoints=len(point_labels))

    @classmethod
    def load(cls, loc: 'DataLocation'):

        storage = storage_factory(loc.storage_type, loc.bucket_name)
        stream = storage.load(loc.key)
        stream.seek(0)
        try:
            data = np.load(stream)
            valid_rowcol = data['meta'][0]
            if valid_rowcol:
                return ImageFeatures(
                    point_features=[PointFeatures(
                        row=int(row),
                        col=int(col),
                        data=feat) for row, col, feat in zip(data['rows'],
                                                             data['cols'],
                                                             data['feat'])],
                    valid_rowcol=bool(data['meta'][0]),
                    npoints=data['meta'][1],
                    feature_dim=data['meta'][2])
            else:
                return cls.deserialize(data['feat'].tolist())

        except ValueError:
            "We used to store these as a JSON file with a list of features."
            data = json.loads(stream.getvalue().decode('utf-8'))
            return cls.deserialize(data)

    def store(self, loc: 'DataLocation'):
        storage = storage_factory(loc.storage_type, loc.bucket_name)
        if self.valid_rowcol:
            rows = np.array(
                [p.row for p in self.point_features], dtype=np.uint16)
            cols = np.array(
                [p.col for p in self.point_features], dtype=np.uint16)
        else:
            rows = np.array([])
            cols = np.array([])
        feat = np.array([p.data for p in self.point_features], dtype=float)
        meta = np.array([self.valid_rowcol, self.npoints, self.feature_dim])
        output = BytesIO()
        np.savez_compressed(output, meta=meta, rows=rows, cols=cols, feat=feat)
        output.seek(0)
        storage.store(loc.key, output)


class ValResults(DataClass):
    """ Defines the validation results data class. Note that the gt and est
    lists points to the index into the classes list."""

    def __init__(self,
                 scores: list[float],
                 gt: list[int],  # Using singular for backwards compatibility.
                 est: list[int],  # Using singular for backwards compatibility.
                 classes: list[int]):

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
    def deserialize(cls, data: dict) -> 'ValResults':
        """ Redefined here to help the Typing module. """
        return ValResults(**data)
