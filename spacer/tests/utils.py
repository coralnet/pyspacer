from spacer import config
from spacer.messages import DataLocation


def cn_beta_fixture_location(key):
    return DataLocation(
        storage_type='s3',
        bucket_name=config.TEST_BUCKET,
        key='legacy_compat/coralnet_beta/' + key
    )
