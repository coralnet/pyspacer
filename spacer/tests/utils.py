# Utilities for unit tests.

from contextlib import contextmanager
import os
import tempfile

from spacer import config
from spacer.messages import DataLocation


def cn_beta_fixture_location(key):
    return DataLocation(
        storage_type='s3',
        bucket_name=config.TEST_BUCKET,
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

    if os.path.exists(temporary_file.name):
        os.remove(temporary_file.name)
