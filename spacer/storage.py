"""
Defines storage ABC; implementations; and factory.
"""

import abc
import os
import pickle
import wget
from PIL import Image
from io import BytesIO
from typing import Union, Tuple

from urllib.error import URLError

from sklearn.calibration import CalibratedClassifierCV, LabelEncoder

from spacer import config

from spacer.messages import DataLocation


def patch_legacy(clf: CalibratedClassifierCV) -> CalibratedClassifierCV:
    """
    Upgrades models trained on sklearn 0.17.1 to 0.22.2
    Note: this in only tested for inference.
    """
    print("-> Patching legacy classifier.")
    assert len(clf.calibrated_classifiers_) == 1
    assert all(clf.classes_ == clf.calibrated_classifiers_[0].classes_)
    clf.calibrated_classifiers_[0].label_encoder_ = LabelEncoder()
    clf.calibrated_classifiers_[0].label_encoder_.fit(clf.classes_)
    return clf


class Storage(abc.ABC):  # pragma: no cover

    @abc.abstractmethod
    def store_classifier(self, path: str, clf: CalibratedClassifierCV) -> None:
        """ Stores a classifier instance """

    @abc.abstractmethod
    def load_classifier(self, path: str) -> CalibratedClassifierCV:
        """ Loads a classifier instance """

    @abc.abstractmethod
    def store_image(self, path: str, content: Image) -> None:
        """ Stores a PIL image instance """

    @abc.abstractmethod
    def load_image(self, path: str) -> Image:
        """ Loads a PIL image instance """

    @abc.abstractmethod
    def store_string(self, path: str, content: str) -> None:
        """ Stores a string """

    @abc.abstractmethod
    def load_string(self, path: str) -> str:
        """ Loads a string """

    @abc.abstractmethod
    def delete(self, path: str) -> None:
        """ Deletes the file if it exists """

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        """ Checks if file exists """


class URLStorage(Storage):
    """ Loads items from URLs. Does not support store operations. """

    def __init__(self):
        self.fs_storage = FileSystemStorage()

    def _load(self, url, local_load_method):
        tmp_path = wget.download(url)
        item = local_load_method(tmp_path)
        self.fs_storage.delete(tmp_path)
        return item

    def load_image(self, url: str):
        return self._load(url, self.fs_storage.load_image)

    def load_classifier(self, url: str):
        return self._load(url, self.fs_storage.load_classifier)

    def load_string(self, url: str):
        return self._load(url, self.fs_storage.load_string)

    def store_image(self, path: str, content: Image):
        raise TypeError('Store operation not supported for URL storage.')

    def store_classifier(self, path: str, clf: CalibratedClassifierCV):
        raise TypeError('Store operation not supported for URL storage.')

    def store_string(self, path: str, content: str):
        raise TypeError('Store operation not supported for URL storage.')

    def delete(self, url: str) -> None:
        raise TypeError('Store operation not supported for URL storage.')

    def exists(self, url: str) -> bool:
        try:
            wget.download(url)
        except URLError:
            return False
        except ValueError:
            return False
        return True


class S3Storage(Storage):
    """ Stores objects on AWS S3 """

    def __init__(self, bucketname: str):
        conn = config.get_s3_conn()
        self.bucket = conn.get_bucket(bucketname)

    def load_classifier(self, path: str):
        key = self.bucket.get_key(path)

        # Make sure pickle.loads is compatible with legacy classifiers which
        # were stored using pickle.dumps in python 2.7.
        clf = pickle.loads(key.get_contents_as_string(), fix_imports=True,
                           encoding='latin1')
        if hasattr(clf, 'calibrated_classifiers_') and not \
                hasattr(clf.calibrated_classifiers_[0], 'label_encoder'):
            clf = patch_legacy(clf)
        return clf

    def store_classifier(self, path: str, clf: CalibratedClassifierCV):
        key = self.bucket.new_key(path)
        key.set_contents_from_string(pickle.dumps(clf, protocol=2))

    def store_image(self, path: str, content: Image):

        with BytesIO() as stream:
            content.save(stream, 'JPEG')
            stream.seek(0)
            key = self.bucket.new_key(path)
            key.set_contents_from_file(stream)

    def load_image(self, path) -> Image:
        key = self.bucket.get_key(path)
        return Image.open(BytesIO(key.get_contents_as_string()))

    def store_string(self, path: str, content: str):
        key = self.bucket.new_key(path)
        key.set_contents_from_string(content)

    def load_string(self, path: str) -> str:
        key = self.bucket.get_key(path)
        return key.get_contents_as_string().decode('UTF-8')

    def delete(self, path: str):
        self.bucket.delete_key(path)

    def exists(self, path: str):
        return self.bucket.get_key(path) is not None


class FileSystemStorage(Storage):
    """ Stores objects on disk """

    def __init__(self):
        pass

    def load_classifier(self, path: str) -> CalibratedClassifierCV:
        with open(path, 'rb') as f:
            clf = pickle.load(f, encoding='latin1')

        if hasattr(clf, 'calibrated_classifiers_') and not \
                hasattr(clf.calibrated_classifiers_[0], 'label_encoder'):
            clf = patch_legacy(clf)
        return clf

    def store_classifier(self, path: str, clf: CalibratedClassifierCV):
        with open(path, 'wb') as f:
            pickle.dump(clf, f, protocol=2)

    def store_image(self, path: str, content: Image):
        content.save(path)

    def load_image(self, path) -> Image:
        return Image.open(path)

    def store_string(self, path: str, content: str):
        with open(path, 'w') as f:
            f.write(content)

    def load_string(self, path: str):
        with open(path, 'r') as f:
            return f.read()

    def delete(self, path: str):
        os.remove(path)

    def exists(self, path: str):
        return os.path.exists(path)


class MemoryStorage(Storage):
    """ This stores objects in RAM. Useful for testing only. """

    def __init__(self):
        self.blobs = {}

    def load_classifier(self, path: str):
        return self.blobs[path]

    def store_classifier(self, path: str, clf: CalibratedClassifierCV):
        self.blobs[path] = clf

    def store_image(self, path: str, content: Image):
        self.blobs[path] = content

    def load_image(self, path: str):
        return self.blobs[path]

    def store_string(self, path: str, content: str):
        self.blobs[path] = content

    def load_string(self, path: str):
        return self.blobs[path]

    def delete(self, path: str):
        del self.blobs[path]

    def exists(self, path: str):
        return path in self.blobs


def storage_factory(storage_type: str, bucketname: Union[str, None] = None):

    assert storage_type in config.STORAGE_TYPES

    if storage_type == 's3':
        print("-> Initializing s3 storage")
        return S3Storage(bucketname=bucketname)
    if storage_type == 'filesystem':
        print("-> Initializing filesystem storage")
        return FileSystemStorage()
    if storage_type == 'memory':
        print("-> Initializing memory storage")
        return MemoryStorage()
    if storage_type == 'url':
        print("-> Initializing URL storage")
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


def load(loc: DataLocation, data_type: str):
    """ Helper method to load any data type from a DataLocation """

    storage = storage_factory(loc.storage_type, loc.bucket_name)
    if data_type == 'image':
        return storage.load_image(loc.key)
    if data_type == 'clf':
        return storage.load_classifier(loc.key)
    if data_type == 'string':
        return storage.load_string(loc.key)
    else:
        raise ValueError('data_type {} not recognized'.format(data_type))


def store(loc: DataLocation,
          content: Union[Image.Image, CalibratedClassifierCV, str],
          data_type: str):
    """ Helper method to store any data type to a DataLocation """

    storage = storage_factory(loc.storage_type, loc.bucket_name)
    if data_type == 'image':
        return storage.store_image(loc.key, content)
    if data_type == 'clf':
        return storage.store_classifier(loc.key, content)
    if data_type == 'string':
        return storage.store_string(loc.key, content)
    else:
        raise ValueError('data_type {} not recognized'.format(data_type))