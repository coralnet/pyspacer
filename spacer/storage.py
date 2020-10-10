"""
Defines storage ABC; implementations; and factory.
"""

import abc
import logging
import os
import pickle
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Union, Tuple
from urllib.error import URLError

import boto3
import botocore
import wget
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV

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

    def store(self, url: str, stream: BytesIO):
        raise TypeError('Store operation not supported for URL storage.')

    def load(self, url: str):
        tmp_path = wget.download(url)
        stream = self.fs_storage.load(tmp_path)
        self.fs_storage.delete(tmp_path)
        return stream

    def delete(self, url: str) -> None:
        raise TypeError('Delete operation not supported for URL storage.')

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
        self.bucketname = bucketname

    def store(self, key: str, stream: BytesIO):
        client = boto3.client('s3')
        client.put_object(Body=stream, Bucket=self.bucketname, Key=key)

    def load(self, key: str):
        s3 = config.get_s3_conn()
        stream = BytesIO()
        s3.Object(self.bucketname, key).download_fileobj(stream)
        return stream

    def delete(self, key: str) -> None:
        s3 = config.get_s3_conn()
        s3.Object(self.bucketname, key).delete()

    def exists(self, key: str):
        s3 = config.get_s3_conn()
        try:
            s3.Object(self.bucketname, key).load()
            return True
        except botocore.exceptions.ClientError as e:
            return False


class FileSystemStorage(Storage):
    """ Stores objects on disk """

    def __init__(self):
        pass

    def store(self, key: str, stream: BytesIO):

        dirname = os.path.dirname(os.path.abspath(key))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
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


def clear_memory_storage():
    """ Clears global memory storage"""
    global _memorystorage
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
    assert config.HAS_LOCAL_MODEL_PATH, "Model path not set or is invalid."
    destination = os.path.join(config.LOCAL_MODEL_PATH, keyname)

    logging.info('Fetching model at {}...'.format(destination))
    if not os.path.isfile(destination):
        logging.info("Downloading {}...".format(keyname))
        s3 = config.get_s3_conn()
        s3.Bucket(config.MODELS_BUCKET).download_file(keyname, destination)
        was_cashed = False
        logging.info("Done downloading {}.".format(keyname))
    else:
        # Already cached, no need to download
        was_cashed = True
    logging.info('Model at {} fetched, was_cashed: {}.'.format(
        destination, was_cashed))
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

    # This warning is due to the sklearn 0.17.1, 0.22.2 migration.
    warnings.filterwarnings('ignore', category=UserWarning,
                            message="Trying to unpickle.*")

    # This future warning also has to do with the unpickling of the
    # legacy model. It uses a to-be-deprecated import statement.
    warnings.filterwarnings('ignore', category=FutureWarning,
                            message="The sklearn.linear_model.*")

    def patch_legacy():
        """
        Upgrades models trained on scikit-learn 0.17.1 to 0.22.2
        Note: this in only tested for inference.
        """
        from sklearn.calibration import LabelEncoder

        logging.info("Patching legacy classifier.")
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


