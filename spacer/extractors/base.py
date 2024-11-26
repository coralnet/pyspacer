"""
Defines feature-extractor ABC; implementations; and factory.
"""

from __future__ import annotations
import abc
import hashlib
import random
import time
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Type

from PIL import Image

from spacer import config
from spacer.data_classes import PointFeatures, ImageFeatures
from spacer.exceptions import ConfigError, HashMismatchError
from spacer.extractors.utils import crop_patches
from spacer.messages import DataLocation, ExtractFeaturesReturnMsg
from spacer.storage import storage_factory


class FeatureExtractor(abc.ABC):

    # Subclasses should define their expected data_locations keys here.
    # See __init__() for an explanation of data_locations.
    DATA_LOCATION_KEYS: list[str] = []
    # Width and height of cropped patches.
    CROP_SIZE = 224

    def __init__(self,
                 data_locations: dict[str, DataLocation],
                 data_hashes: dict[str, str] = None,
                 **kwargs):
        # DataLocations of files/blobs used to load the extractor.
        # Each subclass of FeatureExtractor should expect specific keys
        # to be present in this dict, one key per file/blob.
        self.data_locations = data_locations
        for key in self.DATA_LOCATION_KEYS:
            if key not in data_locations:
                raise ValueError(
                    f"{key} must be present in data_locations for"
                    f" {self.__class__.__name__}.")

        # SHA256 hashes checking the integrity of the extractor
        # files/blobs. The dict keys should match those of data_locations.
        # Don't have to specify a hash for every data location; only the
        # ones you want to check the integrity of.
        self.data_hashes = data_hashes or dict()

    def __call__(self,
                 im: Image,
                 rowcols: list[tuple[int, int]]) \
            -> tuple[ImageFeatures, ExtractFeaturesReturnMsg]:
        """
        Runs feature extraction on a single image. rowcols specifies the
        pixel locations at which to crop square patches, and then features
        are computed for each patch.
        """
        start_time = time.time()

        # Crop patches
        with config.log_entry_and_exit('cropping of {} patches'.format(
                len(rowcols))):
            patch_list = crop_patches(im, rowcols, self.CROP_SIZE)
        del im

        feats, extractor_loaded_remotely = self.patches_to_features(
            patch_list)

        image_features = ImageFeatures(
            point_features=[
                PointFeatures(row=rc[0], col=rc[1], data=ft)
                for rc, ft in zip(rowcols, feats)
            ],
            valid_rowcol=True,
            feature_dim=len(feats[0]),
            npoints=len(feats),
        )
        return_msg = ExtractFeaturesReturnMsg(
            extractor_loaded_remotely=extractor_loaded_remotely,
            runtime=time.time() - start_time,
        )
        return image_features, return_msg

    def patches_to_features(
            self, patch_list: list[Image]) -> tuple[list, bool]:
        """
        Extract features from cropped patches.
        :param patch_list: a list of square images of size CROP_SIZE
        :return: list of extracted features, one per patch; and a bool
          saying whether or not the extractor params were loaded from a
          remote source.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def feature_dim(self) -> int:
        """ Returns the feature dimension of extractor. """

    def serialize(self) -> dict:
        cls = self.__class__
        return dict(
            # Dotted path to the FeatureExtractor subclass,
            # e.g. 'spacer.extract_features.EfficientNetExtractor'
            class_path=f'{cls.__module__}.{cls.__name__}',
            data_locations=dict([
                (key, loc.serialize())
                for key, loc in self.data_locations.items()
            ]),
            data_hashes=self.data_hashes,
        )

    @staticmethod
    def deserialize(data: dict) -> 'FeatureExtractor':
        working_data = data.copy()
        class_path = working_data.pop('class_path')

        module_path, class_name = class_path.rsplit('.', 1)
        module = import_module(module_path)
        extractor_class: Type['FeatureExtractor'] = getattr(module, class_name)

        working_data['data_locations'] = dict([
            (key, DataLocation.deserialize(serialized_loc))
            for key, serialized_loc in working_data['data_locations'].items()
        ])

        return extractor_class(**working_data)

    def __repr__(self):
        """
        Seeing the serialized fields tends to be a more useful repr than
        `<SomeExtractor object>`.
        """
        return str(self.serialize())

    def __eq__(self, other):
        """This is for some tests comparing equality of messages."""
        return self.serialize() == other.serialize()

    def _check_data_hash(self, data, key):
        if key in self.data_hashes:
            sha256 = hashlib.sha256(data).hexdigest()
            if sha256 != self.data_hashes[key]:
                raise HashMismatchError(
                    f"Hash doesn't match for extractor's"
                    f" '{key}' file/blob.")

    def data_filepath_for_cache(self, key: str) -> Path:
        if not config.EXTRACTORS_CACHE_DIR:
            raise ConfigError(
                "Must define EXTRACTORS_CACHE_DIR"
                " to load extractor files/blobs into the filesystem.")

        loc = self.data_locations[key]
        return Path(config.EXTRACTORS_CACHE_DIR) / loc.filename

    def decache_remote_loaded_file(self, key: str):
        loc = self.data_locations[key]
        if not loc.is_remote:
            return

        file_storage = storage_factory('filesystem')
        filepath_for_cache = str(self.data_filepath_for_cache(key))
        if file_storage.exists(filepath_for_cache):
            file_storage.delete(filepath_for_cache)

    def load_data_into_filesystem(self, key: str) -> tuple[str, bool]:
        """
        Ensure the data file/blob of the given key is loaded into the
        filesystem.
        Returns:
        - A filesystem path for that data file/blob.
        - True if it had to be loaded from remote storage, else False.
        """
        loc = self.data_locations[key]

        if loc.storage_type == 'filesystem':
            return loc.key, False

        filepath_for_cache = self.data_filepath_for_cache(key)
        file_storage = storage_factory('filesystem')
        already_in_filesystem = filepath_for_cache.exists()

        if not already_in_filesystem:
            # Must load into the filesystem.
            storage = storage_factory(loc.storage_type, loc.bucket_name)
            data = storage.load(loc.key)
            file_storage.store(str(filepath_for_cache), data)

        remote_loaded = not already_in_filesystem and loc.is_remote
        return str(filepath_for_cache), remote_loaded

    def load_datastream(self, key: str) -> tuple[BytesIO, bool]:
        """
        Loads extractor data from the DataLocation
        `self.data_locations[key]`.
        Returns:
        - Byte stream of the data file/blob.
        - True if it had to be loaded from remote storage, else False.

        If the location is remote, caches the data to the filesystem.
        Cached files are identified by the source location's filename. So
        be sure that you don't have distinct files with the same filename,
        or they'll collide with each other.

        Windows beware: behavior is undefined when there are distinct files
        to cache whose filenames differ only in upper/lowercase.
        """
        loc = self.data_locations[key]

        if loc.is_remote:

            data_filepath, remote_loaded = self.load_data_into_filesystem(key)
            file_storage = storage_factory('filesystem')
            datastream = file_storage.load(data_filepath)

            if remote_loaded:
                # Cache miss.
                check_hash = True
            else:
                # Cache hit.
                # Skip the hash check in this case, since presumably the
                # hash was already checked when loading into the cache, and
                # hash checking for large files may be time consuming.
                check_hash = False

        else:

            storage = storage_factory(loc.storage_type, loc.bucket_name)
            datastream = storage.load(loc.key)

            remote_loaded = False
            check_hash = True

        if check_hash:
            data = datastream.read()
            try:
                self._check_data_hash(data, key)
            except HashMismatchError as e:
                # If the hash doesn't match, we don't want to proceed
                # as if the extractor is OK to use.
                # For the remote case, that means we must decache,
                # because loading from cache next time would allow the
                # hash check to be skipped.
                if loc.is_remote:
                    self.decache_remote_loaded_file(key)
                raise e
            # Reset the stream so another function (like torch.load()) can
            # read it later.
            datastream.seek(0)

        return datastream, remote_loaded


class DummyExtractor(FeatureExtractor):
    """
    This doesn't actually extract any features from the image;
    it just returns dummy information in the correct format.
    """
    def __init__(self,
                 data_locations: dict[str, DataLocation] = None,
                 feature_dim: int = 4096,
                 **kwargs):
        super().__init__(data_locations or dict(), **kwargs)

        # If you want these features to be compatible with an actual
        # classifier, you should make this match the feature_dim that
        # the classifier was trained on.
        self._feature_dim = feature_dim

    def __call__(self, im, rowcols):
        return ImageFeatures(
            point_features=[
                PointFeatures(
                    row=rc[0],
                    col=rc[1],
                    data=[random.random() for _ in range(self.feature_dim)],
                )
                for rc in rowcols
            ],
            valid_rowcol=True,
            feature_dim=self.feature_dim,
            npoints=len(rowcols),
        ), ExtractFeaturesReturnMsg.example()

    @property
    def feature_dim(self):
        return self._feature_dim

    def serialize(self) -> dict:
        data = super().serialize()
        data['feature_dim'] = self.feature_dim
        return data


class VGG16CaffeExtractor(FeatureExtractor):

    # definition should be a Caffe prototxt file, typically .prototxt
    # weights should be a Caffe model file, typically .caffemodel
    DATA_LOCATION_KEYS = ['definition', 'weights']

    def __call__(self, im, rowcols):
        if not config.HAS_CAFFE:
            raise ConfigError(
                f"Need Caffe installed to call"
                f" {self.__class__.__name__}.")

        return super().__call__(im, rowcols)

    def patches_to_features(self, patch_list):
        # We should only reach this line if it is confirmed caffe is available
        from spacer.extractors.vgg16 import classify_from_patchlist

        # Set caffe parameters
        caffe_params = {'im_mean': [128, 128, 128],
                        'scaling_method': 'scale',
                        'batch_size': 10}

        # Extract features
        definition_filepath, _ = (
            self.load_data_into_filesystem('definition'))
        weights_filepath, extractor_loaded_remotely = (
            self.load_data_into_filesystem('weights'))

        features = classify_from_patchlist(
            patch_list,
            caffe_params,
            definition_filepath,
            weights_filepath,
            scorelayer='fc7',
        )
        features = [feat.tolist() for feat in features]
        return features, extractor_loaded_remotely

    @property
    def feature_dim(self):
        return 4096
