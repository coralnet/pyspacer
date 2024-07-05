"""
Defines storage ABC; implementations; and factory.
"""

from __future__ import annotations
import abc
import os
import pickle
from functools import lru_cache
from http.client import IncompleteRead
from io import BytesIO
from pathlib import Path
from pickle import Unpickler
from urllib.error import URLError
import urllib.request

import botocore.exceptions
from boto3.s3.transfer import TransferConfig
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV

from spacer import config
from spacer.exceptions import URLDownloadError


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


class RemoteStorage(Storage):
    """
    Base class for types of storage which typically involve going out to
    the network.
    """

    def load(self, key: str, filesystem_cache: str | None = None):
        filesystem_storage = storage_factory('filesystem')

        if not filesystem_cache:
            # Load from remote.
            return self._load_remote(key)

        # From this point on, there's a filesystem cache.

        cache_filepath = str(Path(filesystem_cache, key))

        if filesystem_storage.exists(cache_filepath):
            # Load file from the cache.
            loaded_stream = filesystem_storage.load(cache_filepath)
        else:
            # Didn't load from cache; load from remote.
            loaded_stream = self._load_remote(key)

            # Cache loaded file to the provided filesystem dir.
            loaded_stream.seek(0)
            filesystem_storage.store(cache_filepath, loaded_stream)

        return loaded_stream

    @abc.abstractmethod
    def _load_remote(self, key: str):
        pass


class URLStorage(RemoteStorage):
    """ Loads items from URLs. Does not support store operations. """

    TIMEOUT = 20.0

    def __init__(self):
        self.fs_storage = FileSystemStorage()

    def store(self, url: str, stream: BytesIO):
        raise TypeError('Store operation not supported for URL storage.')

    def _load_remote(self, url: str) -> BytesIO:
        try:
            # The timeout here defines the max time for both:
            # - The initial connection; else URLError - <urlopen error timed
            #   out> is raised.
            # - A single idle period mid-response (not the entire response
            #   time); else TimeoutError - timed out is raised.
            download_response = urllib.request.urlopen(
                url, timeout=self.TIMEOUT)
        except (TimeoutError, URLError, ValueError) as e:
            # Besides timeouts, possible errors include:
            # ValueError: unknown url type: '<url>'
            #   - Malformed url
            # URLError: <urlopen error [Errno -5] No address associated with
            # hostname> [Linux]
            # OR gaierror(11001, 'getaddrinfo failed') [Win]
            #   - Invalid domain
            # URLError: <urlopen error [Errno -3] Temporary failure in name
            # resolution>
            #   - No internet
            # HTTPError 404 or 500
            #   - HTTPError inherits from URLError
            raise URLDownloadError(
                f"Failed to download from the URL '{url}'.", e)

        try:
            download_bytes = download_response.read()
        except IncompleteRead as e:
            # http.client.IncompleteRead - <num> bytes read, <num> more expected
            #   - In some cases this seems to just happen randomly?
            #     But it'll depend on the URL's server.
            raise URLDownloadError(
                f"Couldn't read the full response from the URL '{url}'.", e)

        return BytesIO(download_bytes)

    def delete(self, url: str) -> None:
        raise TypeError('Delete operation not supported for URL storage.')

    def exists(self, url: str) -> bool:
        """
        URL existence can be a hard problem.
        This implementation makes no guarantees on correctness.
        """
        # HEAD can check for existence without downloading the entire resource
        try:
            request = urllib.request.Request(url, method='HEAD')
        except ValueError:
            # Might be an invalid URL format
            return False

        try:
            urllib.request.urlopen(request, timeout=self.TIMEOUT)
        except (TimeoutError, URLError):
            # Might be an unreachable domain, 404, or something else
            return False
        return True


class S3Storage(RemoteStorage):
    """ Stores objects on AWS S3 """

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        # Prevent `RuntimeError: cannot schedule new futures after
        # interpreter shutdown`.
        # Based on https://github.com/etianen/django-s3-storage/pull/136
        self.transfer_config = TransferConfig(use_threads=False)

    def store(self, key: str, stream: BytesIO):
        s3 = config.get_s3_conn()
        s3.Bucket(self.bucket_name).put_object(Body=stream, Key=key)

    def _load_remote(self, key: str):
        s3 = config.get_s3_conn()
        stream = BytesIO()
        s3.Object(self.bucket_name, key).download_fileobj(
            stream, Config=self.transfer_config)
        return stream

    def delete(self, key: str) -> None:
        s3 = config.get_s3_conn()
        s3.Object(self.bucket_name, key).delete()

    def exists(self, key: str):
        s3 = config.get_s3_conn()
        try:
            s3.Object(self.bucket_name, key).load()
            return True
        except botocore.exceptions.ClientError:
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


def storage_factory(storage_type: str, bucket_name: str | None = None):

    assert storage_type in config.STORAGE_TYPES

    if storage_type == 's3':
        if bucket_name is None:
            raise ValueError("bucket_name must be a string for s3 storage")
        return S3Storage(bucket_name=bucket_name)
    if storage_type == 'filesystem':
        return FileSystemStorage()
    if storage_type == 'memory':
        global _memorystorage
        if _memorystorage is None:
            _memorystorage = MemoryStorage()
        return _memorystorage
    if storage_type == 'url':
        return URLStorage()


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
    if not hasattr(clf, 'calibrated_classifiers_'):
        raise ValueError("Only fitted classifiers can be stored.")
    storage = storage_factory(loc.storage_type, loc.bucket_name)
    storage.store(loc.key, BytesIO(pickle.dumps(clf, protocol=2)))


class ClassifierUnpickler(Unpickler):
    def load(self):
        clf = super().load()

        if not isinstance(clf, CalibratedClassifierCV):
            raise ValueError(
                f"Loaded a {type(clf).__name__}"
                f" instead of a CalibratedClassifierCV.")

        return clf


@lru_cache(maxsize=3)
def load_classifier(loc: 'DataLocation'):

    storage = storage_factory(loc.storage_type, loc.bucket_name)
    stream = storage.load(loc.key)
    stream.seek(0)

    clf = ClassifierUnpickler(
        stream, fix_imports=True, encoding='latin1').load()

    return clf
