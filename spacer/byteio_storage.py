import abc

from io import BytesIO

from spacer import config


class Storage(abc.ABC):  # pragma: no cover

    @abc.abstractmethod
    def load(self, loc: 'DataLocation') -> BytesIO:
        pass

    @abc.abstractmethod
    def store(self, loc: 'DataLocation', content: BytesIO):
        pass


class S3Storage(Storage):

    def load(self, loc):

        conn = config.get_s3_conn()
        bucket = conn.get_bucket(loc.bucket_name)
        key = bucket.get_key(loc.key)
        return BytesIO(key.get_contents_as_string())

    def store(self, loc, stream: BytesIO):
        conn = config.get_s3_conn()
        bucket = conn.get_bucket(loc.bucket_name)
        key = bucket.new_key(loc.key)
        key.set_contents_from_file(stream)


class FileSystemStorage(Storage):

    def load(self, loc):
        with open(loc.key, 'rb') as f:
            return BytesIO(f.read())

    def store(self, loc: 'DataLocation', content: BytesIO):
        with open(loc.key, 'wb') as f:
            f.write(content.getbuffer())
