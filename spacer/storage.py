"""
Defines storage ABC; implementations; and factory.
"""

import abc
import os
import pickle
import wget
from functools import lru_cache
from PIL import Image
from io import BytesIO
from typing import Union, Tuple

from urllib.error import URLError

from sklearn.calibration import CalibratedClassifierCV, LabelEncoder

from spacer import config


class Storage(abc.ABC):  # pragma: no cover

    @abc.abstractmethod
    def store(self, key: str, stream: BytesIO) -> None:
        """ Stores a BytesIO stream """

    @abc.abstractmethod
    def load(self, key: str) -> BytesIO:
        """ Loads key to BytesIO stream """

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """ Deletes key if it exists """

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """ Checks if key exists """


class URLStorage(Storage):
    """ Loads items from URLs. Does not support store operations. """

    def __init__(self):
        self.fs_storage = FileSystemStorage()

    def _load(self, url, local_load_method):
        tmp_path = wget.download(url)
        item = local_load_method(tmp_path)
        self.fs_storage.delete(tmp_path)
        return item

    def store(self, url: str, stream: BytesIO):
        raise TypeError('Store operation not supported for URL storage.')

    def load(self, url: str):
        tmp_path = wget.download(url)
        stream = self.fs_storage.load(tmp_path)
        self.fs_storage.delete(tmp_path)
        return stream

    def delete(self, url: str) -> None:
        raise TypeError('Store operation not supported for URL storage.')

    def exists(self, url: str) -> bool:
        try:
            tmp_path = wget.download(url)
        except URLError:
            return False
        except ValueError:
            return False
        self.fs_storage.delete(tmp_path)
        return True


class S3Storage(Storage):
    """ Stores objects on AWS S3 """

    def __init__(self, bucketname: str):
        conn = config.get_s3_conn()
        self.bucket = conn.get_bucket(bucketname)

    def store(self, key: str, stream: BytesIO):
        key = self.bucket.new_key(key)
        key.set_contents_from_file(stream)

    def load(self, key: str):
        key = self.bucket.get_key(key)
        return BytesIO(key.get_contents_as_string())

    def delete(self, key: str) -> None:
        self.bucket.delete_key(key)

    def exists(self, key: str):
        return self.bucket.get_key(key) is not None


class FileSystemStorage(Storage):
    """ Stores objects on disk """

    def __init__(self):
        pass

    def store(self, key: str, stream: BytesIO):
        with open(key, 'wb') as f:
            f.write(stream.getbuffer())

    def load(self, key: str):
        with open(key, 'rb') as f:
            return BytesIO(f.read())

    def delete(self, path: str):
        os.remove(path)

    def exists(self, path: str):
        return os.path.exists(path)


class MemoryStorage(Storage):
    """ This stores objects in RAM. Useful for testing only. """

    def __init__(self):
        self.blobs = {}

    def store(self, key: str, stream: BytesIO):
        self.blobs[key] = stream.getvalue()

    def load(self, key: str):
        stream = BytesIO(self.blobs[key])
        return stream

    def delete(self, path: str):
        del self.blobs[path]

    def exists(self, path: str):
        return path in self.blobs


# This holds the global memory storage.
_memorystorage = None


def storage_factory(storage_type: str, bucketname: Union[str, None] = None):

    assert storage_type in config.STORAGE_TYPES

    if storage_type == 's3':
        return S3Storage(bucketname=bucketname)
    if storage_type == 'filesystem':
        return FileSystemStorage()
    if storage_type == 'memory':
        global _memorystorage
        if _memorystorage is None:
            _memorystorage = MemoryStorage()
        return _memorystorage
    if storage_type == 'url':
        return URLStorage()


def download_model(keyname: str) -> Tuple[str, bool]:
    """
    Utility method to download model to local cache.
    This is not part of the storage interface since it is a special case,
    and models need to be downloaded to a specific destination folder to be
    shared with host filesystem.
    """
    assert config.HAS_S3_MODEL_ACCESS, "Need access to model bucket."
    destination = os.path.join(config.LOCAL_MODEL_PATH, keyname)
    if not os.path.isfile(destination):
        print("-> Downloading {}".format(keyname))
        conn = config.get_s3_conn()
        bucket = conn.get_bucket(config.MODELS_BUCKET, validate=True)
        key = bucket.get_key(keyname)
        key.get_contents_to_filename(destination)
        was_cashed = False
    else:
        # Already cached, no need to download
        was_cashed = True

    return destination, was_cashed


def store_image(loc: 'DataLocation', img: Image):
    storage = storage_factory(loc.storage_type, loc.bucket_name)
    with BytesIO() as stream:
        img.save(stream, 'JPEG')
        stream.seek(0)
        storage.store(loc.key, stream)


def load_image(loc: 'DataLocation'):
    storage = storage_factory(loc.storage_type, loc.bucket_name)
    return Image.open(storage.load(loc.key))


def store_classifier(loc: 'DataLocation', clf: CalibratedClassifierCV):
    storage = storage_factory(loc.storage_type, loc.bucket_name)
    storage.store(loc.key, BytesIO(pickle.dumps(clf, protocol=2)))


@lru_cache(maxsize=3)
def load_classifier(loc: 'DataLocation'):

    def patch_legacy():
        """
        Upgrades models trained on scikit-learn 0.17.1 to 0.22.2
        Note: this in only tested for inference.
        """
        print("-> Patching legacy classifier.")
        assert len(clf.calibrated_classifiers_) == 1
        assert all(clf.classes_ == clf.calibrated_classifiers_[0].classes_)
        clf.calibrated_classifiers_[0].label_encoder_ = LabelEncoder()
        clf.calibrated_classifiers_[0].label_encoder_.fit(clf.classes_)

    storage = storage_factory(loc.storage_type, loc.bucket_name)
    clf = pickle.loads(storage.load(loc.key).getbuffer(), fix_imports=True,
                       encoding='latin1')

    if hasattr(clf, 'calibrated_classifiers_') and not \
            hasattr(clf.calibrated_classifiers_[0], 'label_encoder'):
        patch_legacy()

    return clf


