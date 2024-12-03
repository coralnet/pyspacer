# Utilities for unit tests.

from contextlib import contextmanager
import tempfile
import uuid

import numpy as np
from PIL import Image

from spacer import config
from spacer.messages import DataLocation
from spacer.storage import FileSystemStorage, S3Storage


def cn_beta_fixture_location(key):
    return DataLocation(
        storage_type='s3',
        bucket_name=config.CN_FIXTURES_BUCKET,
        key='legacy_compat/coralnet_beta/' + key
    )


@contextmanager
def temp_filesystem_data_location():
    temporary_file = tempfile.NamedTemporaryFile(delete=False)

    # We assume the file will be reopened, so close it first.
    temporary_file.close()

    temp_data_location = DataLocation(
        storage_type='filesystem',
        key=temporary_file.name,
        bucket_name=''
    )

    yield temp_data_location

    storage = FileSystemStorage()
    if storage.exists(temporary_file.name):
        storage.delete(temporary_file.name)


@contextmanager
def temp_s3_filepaths(
    # Name of S3 bucket to claim filepaths in
    bucket_name: str,
    # How many filepaths to claim
    num_data_locations: int = 1,
):
    # Claim some filepaths in the bucket, for the in-context code to
    # use.
    filenames = []
    for _ in range(num_data_locations):
        random_base_filename = uuid.uuid4()
        filenames.append(f'tmp_{random_base_filename}')

    yield filenames

    # Clean up any files created at those filepaths.
    storage = S3Storage(bucket_name=bucket_name)
    for filename in filenames:
        if storage.exists(filename):
            storage.delete(filename)


def random_image(width, height) -> Image:
    """
    Source: https://stackoverflow.com/a/10901092
    """
    arr = np.random.rand(width, height, 3) * 255
    return Image.fromarray(arr.astype('uint8')).convert('RGB')
