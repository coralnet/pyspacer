"""
Defines data-classes for input and output types.
Each data-class can serialize itself to a structure of JSON-friendly
python-native data-structures such that it can be stored.
"""
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from pprint import pformat
from typing import TypeAlias
from urllib.parse import urlparse

import numpy as np

from spacer import config
from spacer.exceptions import RowColumnMissingError, RowColumnMismatchError
from spacer.storage import RemoteStorage, storage_factory


# LabelId is constrained by scikit-learn's usage of 'targets':
# https://scikit-learn.org/stable/glossary.html#term-target
# However, more types are possible besides int and str (mainly, combinations
# thereof).
LabelId: TypeAlias = int | str
Annotation: TypeAlias = tuple[int, int, LabelId]
FeatureLabelPair: TypeAlias = tuple[np.ndarray, LabelId]
FeatureLabelBatch: TypeAlias = tuple[list[np.ndarray], list[LabelId]]


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
    def example(cls) -> DataLocation:
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
    def deserialize(cls, data: dict) -> DataLocation:
        return DataLocation(**data)

    def serialize(self) -> dict:
        return dict(
            storage_type=self.storage_type,
            key=self.key,
            bucket_name=self.bucket_name,
        )

    # __eq__ and __hash__ allow instances to be used as dict keys.

    def __eq__(self, other):
        return (
            self.storage_type == other.storage_type
            and self.key == other.key
            and self.bucket_name == other.bucket_name
        )

    def __hash__(self):
        return hash((self.storage_type, self.key, self.bucket_name))


class ImageLabels(DataClass):
    """ Contains row, col, label information for a set of images. """

    def __init__(
        self,
        # Tuples of image feature vectors and corresponding Annotations.
        data: dict[DataLocation, list[Annotation]] = None,
    ):
        self._data: dict[DataLocation, list[Annotation]] = dict()
        self.label_count_per_class = Counter()
        self.has_remote_data = False

        if data:
            for feature_loc, single_image_annotations in data.items():
                self.add_image(feature_loc, single_image_annotations)
        # Else, we presume the caller means to call add_image() themselves.

        self.filesystem_cache: str | None = None

    def add_image(
        self,
        feature_loc: DataLocation,
        image_annotations: list[Annotation],
    ):
        self._data[feature_loc] = image_annotations

        labels = [label for (row, col, label) in image_annotations]
        self.label_count_per_class.update(labels)

        if not self.has_remote_data and feature_loc.is_remote:
            self.has_remote_data = True

    def set_filesystem_cache(self, dir_path: str):
        self.filesystem_cache = dir_path

    @property
    def classes_set(self):
        return set(self.label_count_per_class.keys())

    @property
    def label_count(self):
        return self.label_count_per_class.total()

    @classmethod
    def example(cls):
        return cls({
            DataLocation('filesystem', 'img1.features'):
                [(100, 200, 3), (101, 200, 2)],
            DataLocation('filesystem', 'img2.features'):
                [(100, 202, 3), (101, 200, 3)],
            DataLocation('filesystem', 'img3.features'):
                [(100, 202, 3), (101, 200, 3)],
            DataLocation('filesystem', 'img4.features'):
                [(100, 202, 3), (101, 200, 3)],
        })

    def serialize(self) -> dict[str, list[Annotation]]:
        """
        Only need the `_data` field; the other fields can be recomputed.

        For the DataLocations, we call serialize() on them to get dicts,
        then stringify those to get something hashable.
        """
        return {
            json.dumps(loc.serialize()): image_annotations
            for loc, image_annotations in self._data.items()
        }

    @classmethod
    def deserialize(
            cls, data: dict[str, list[Annotation]]) -> ImageLabels:
        """
        Custom deserializer required to convert lists to tuples, and to
        deserialize the DataLocations.
        """
        return cls({
            DataLocation.deserialize(json.loads(serialized_loc)):
                [tuple(anno) for anno in image_annotations]
            for serialized_loc, image_annotations in data.items()
        })

    def keys(self):
        return self._data.keys()

    def filter_classes(self, accepted_classes) -> ImageLabels:
        """
        Make a new instance by filtering out labels not included in
        the specified classes.
        """
        filtered_instance = ImageLabels()
        for feature_loc in self.keys():
            this_image_annotations = [
                (row, column, label)
                for row, column, label in self._data[feature_loc]
                if label in accepted_classes
            ]
            # Only include an image if it has any annotations remaining
            # after filtering.
            if len(this_image_annotations) > 0:
                filtered_instance.add_image(
                    feature_loc, this_image_annotations)
        return filtered_instance

    def load_data_in_batches(
        self,
        # Default batch size is arbitrary. None would mean everything
        # is a single batch.
        batch_size: int | None = 1000,
        # If not None, then the feature vectors are iterated through
        # in random order (as opposed to a fixed order if None).
        # This integer is the randomization seed. Pass the same seed to
        # get repeatable results.
        random_seed: int | None = None,
    ) -> Generator[FeatureLabelBatch, None, None]:
        """
        Loads features and labels; generates batches of
        element-matching pairs, which can go into methods
        such as CalibratedClassifierCV.fit().

        Since this is a generator, it does not have to load all feature
        vectors in memory at the same time; only as many as will fit in
        a batch.
        Of course, what you do with the generated results is also
        important for memory usage.

        Note that a single image's features may straddle multiple batches.
        """
        pairs_buffer = []

        keys = self.keys()

        if random_seed is not None:
            # Shuffle the order of images.
            keys = list(keys)
            np.random.seed(random_seed)
            np.random.shuffle(keys)

        for feature_loc in keys:

            # Load the features.
            features = ImageFeatures.load(feature_loc, self.filesystem_cache)

            # Pair them up with the corresponding annotations.
            image_annotations = self._data[feature_loc]
            try:
                pairs_buffer.extend(
                    features.match_with_annotations(image_annotations))
            except RowColumnMissingError:
                raise RowColumnMissingError(
                    f"{feature_loc.key}: Features without rowcols are no"
                    f" longer supported for training.")
            except RowColumnMismatchError as e:
                raise RowColumnMismatchError(f"{feature_loc.key}: {e}")

            if batch_size:
                while len(pairs_buffer) > batch_size:
                    # Slice to respect batch size.
                    # zip(*...) to go from a list of feature-label pairs
                    # to a pair of lists.
                    yield zip(*(pairs_buffer[:batch_size]))
                    pairs_buffer = pairs_buffer[batch_size:]
            # Else, it's all one batch throughout this function.
            # Careful not to exhaust memory.

        if len(pairs_buffer) > 0:
            # Batch of the last features/labels.
            yield zip(*pairs_buffer)

    def load_all_data(
        self,
        random_seed: int | None = None,
    ) -> FeatureLabelBatch:
        """
        Like load_data_in_batches() with batch size None, meaning
        everything is in one batch. Also just returns the result instead
        of being a generator.
        """
        return next(
            self.load_data_in_batches(
                batch_size=None, random_seed=random_seed)
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, loc: DataLocation):
        return self._data[loc]

    def __contains__(self, loc: DataLocation):
        return loc in self._data


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

    def match_with_annotations(
        self,
        labels_data: list[Annotation],
    ) -> Generator[FeatureLabelPair, None, None]:

        if not self.valid_rowcol:
            # This function no longer supports legacy features, since
            # there aren't any known use cases left.
            raise RowColumnMissingError

        # Check that the sets of row, col
        # given by the labels are available in the features.
        rc_features_set = set([(pf.row, pf.col) for pf in
                               self.point_features])
        rc_labels_set = set([(row, col) for (row, col, _) in labels_data])

        if not rc_labels_set.issubset(rc_features_set):
            difference_set = rc_labels_set.difference(rc_features_set)
            example_rc = next(iter(difference_set))
            raise RowColumnMismatchError(
                f"The labels' row-column positions don't match"
                f" those of the feature vector (example: {example_rc}).")

        for row, col, label in labels_data:
            yield self[(row, col)], label

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
                    point_labels: list[LabelId],
                    feature_dim: int):
        pfs = [
            PointFeatures(
                row=itt,
                col=itt,
                # Generate floats that depend on the input labels.
                # The goal here is to strike a balance between
                # the extremes of
                # "mutually exclusive data for any 2 possible labels" and
                # "data of 2 possible labels is so similar that any trained
                # classifier is no better than random chance".
                data=list(hash(label) + np.random.randn(feature_dim)),
            )
            for itt, label in enumerate(point_labels)
        ]

        return ImageFeatures(point_features=pfs,
                             valid_rowcol=True,
                             feature_dim=feature_dim,
                             npoints=len(point_labels))

    @classmethod
    def load(cls, loc: 'DataLocation', filesystem_cache: str | None = None):
        storage = storage_factory(loc.storage_type, loc.bucket_name)
        if isinstance(storage, RemoteStorage):
            stream = storage.load(loc.key, filesystem_cache)
        else:
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
                 classes: list[LabelId]):

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
