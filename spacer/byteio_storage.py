import abc

from io import BytesIO

from spacer import config

from spacer.data_classes import DataLocation


class Storage(abc.ABC):  # pragma: no cover

    @abc.abstractmethod
    def load(self, loc: DataLocation) -> BytesIO:
        pass

    @abc.abstractmethod
    def store(self, loc: DataLocation, content: BytesIO):
        pass


class S3Storage(Storage):

    def load(self, loc):

        conn = config.get_s3_conn()
        bucket = conn.get_bucket(loc.bucketname)
        key = bucket.get_key(loc.key)
        return BytesIO(key.get_content_as_string())

    def store(self, loc, stream: BytesIO):
        conn = config.get_s3_conn()
        bucket = conn.get_bucket(loc.bucketname)
        key = bucket.new_key(loc.key)
        key.set_content_from_file(stream)


