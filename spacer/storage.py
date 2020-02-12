import abc
import os
import pickle
from io import BytesIO
from typing import Union, Tuple

import boto
from PIL import Image

from spacer import config

from sklearn.calibration import CalibratedClassifierCV


class Storage(abc.ABC):

    @abc.abstractmethod
    def store_classifier(self, path: str, clf: CalibratedClassifierCV) -> None:
        pass

    def load_classifier(self, path: str) -> CalibratedClassifierCV:
        pass

    @abc.abstractmethod
    def load_image(self, path: str) -> Image:
        pass

    @abc.abstractmethod
    def store_string(self, path: str, content: str) -> None:
        pass

    @abc.abstractmethod
    def load_string(self, path: str) -> str:
        pass

    @abc.abstractmethod
    def delete(self, path: str) -> None:
        """ Deletes the file if it exists """
        pass

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        """ Checks if file exists """
        pass


class S3Storage(Storage):

    def __init__(self, bucketname: str):

        conn = boto.connect_s3()
        self.bucket = conn.get_bucket(bucketname)

    def load_classifier(self, path: str):
        key = self.bucket.get_key(path)

        # Make sure pickle.loads is compatible with legacy classifiers which
        # were stored using pickle.dumps in python 2.7.
        return pickle.loads(key.get_contents_as_string(), fix_imports=True,
                            encoding='latin1')

    def store_classifier(self, path: str, clf: CalibratedClassifierCV):
        key = self.bucket.new_key(path)
        key.set_contents_from_string(pickle.dumps(clf, protocol=2))

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


class LocalStorage(Storage):

    def __init__(self):
        pass

    def load_classifier(self, path: str) -> CalibratedClassifierCV:
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')

    def store_classifier(self, path: str, clf: CalibratedClassifierCV):
        with open(path, 'wb') as f:
            pickle.dump(clf, f, protocol=2)

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


def storage_factory(storage_type: str, bucketname: Union[str, None]):

    assert storage_type in config.STORAGE_TYPES

    if storage_type == 's3':
        print("-> Initializing s3 storage")
        return S3Storage(bucketname=bucketname)
    elif storage_type == 'local':
        print("-> Initializing local storage")
        return LocalStorage()
    else:
        raise ValueError('Unknown storage type: {}'.format(storage_type))


def download_model(keyname: str) -> Tuple[str, bool]:
    """ Utility method to download model with to local cache. """

    destination = os.path.join(config.LOCAL_MODEL_PATH, keyname)
    if not os.path.isfile(destination):
        print("-> Downloading {}".format(keyname))
        conn = boto.connect_s3()
        bucket = conn.get_bucket(config.MODELS_BUCKET, validate=True)
        key = bucket.get_key(keyname)
        key.get_contents_to_filename(destination)
        was_cashed = False
    else:
        # Already cached, no need to download
        was_cashed = True

    return destination, was_cashed
