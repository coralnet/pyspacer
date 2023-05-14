"""
Contains config and settings for the repo.
"""

import importlib
import json
import logging
import os
import time
import warnings
from contextlib import ContextDecorator
from pathlib import Path
from typing import Tuple, Optional

import boto3
import botocore.exceptions
from PIL import Image, ImageFile


def filter_warnings():
    """ Filters out some verified warnings. """

    # Per discussion in https://github.com/boto/boto3/issues/454,
    # the boto package is raising a lot of warnings that it shouldn't.
    warnings.filterwarnings("ignore", category=ResourceWarning,
                            message="unclosed.*<ssl.SSLSocket.*>")
    warnings.filterwarnings("ignore", category=ResourceWarning,
                            message="unclosed.*<_io.TextIOWrapper.*>")


APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent


def get_secret(key):
    """ Try to load settings from secrets.json file """
    secrets_path = REPO_DIR / 'secrets.json'
    if not secrets_path.exists():  # pragma: no cover
        RuntimeWarning('secrets.json not found')
        return None
    else:
        try:
            with open(secrets_path) as fp:
                secrets = json.load(fp)
            return secrets[key]
        except Exception as err_:  # pragma: no cover
            RuntimeWarning(
                'Unable to parse secrets.json: {}'.format(repr(err_)))
            return None


def get_config_value(key: str, required: bool) -> Optional[str|Path]:
    # Try environment variables. Each should be prefixed with 'SPACER_'.
    value = os.getenv('SPACER_' + key)
    if value:
        return value

    # Try secrets file. Example:
    # {
    #     "AWS_ACCESS_KEY_ID": "...",
    #     "AWS_SECRET_ACCESS_KEY": "..."
    # }
    value = get_secret(key)
    if value:
        return value

    # Try Django settings. Example:
    # SPACER = {
    #     'AWS_ACCESS_KEY_ID': '...',
    #     'AWS_SECRET_ACCESS_KEY': '...',
    # }
    try:
        from django.conf import settings
        from django.core.exceptions import ImproperlyConfigured

        try:
            spacer_settings: dict = getattr(settings, 'SPACER')
            value = spacer_settings[key]
        except (ImproperlyConfigured, AttributeError, KeyError):
            value = None
    except ImportError:
        value = None

    if required and not value:
        raise RuntimeError(f"{key} setting is required.")

    # Ensure we don't return an empty string
    return value or None


def get_aws_credentials() -> Tuple[Optional[str], Optional[str]]:
    return (
        get_config_value('AWS_ACCESS_KEY_ID', required=False),
        get_config_value('AWS_SECRET_ACCESS_KEY', required=False),
    )


def get_s3_conn():
    """
    Returns a boto s3 connection.
    - It first looks for credentials in spacer config.
    - If not found there it will default to credentials in ~/.aws/credentials
    """
    aws_key_id, aws_key_secret = get_aws_credentials()

    return boto3.resource('s3',
                          region_name="us-west-2",
                          aws_access_key_id=aws_key_id,
                          aws_secret_access_key=aws_key_secret)


class log_entry_and_exit(ContextDecorator):
    def __init__(self, name):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        logging.info('Entering: %s', self.name)

    def __exit__(self, exc_type, exc, exc_tb):
        logging.info('Exiting: %s after %f seconds.', self.name,
                     time.time() - self.start_time)


LOCAL_MODEL_PATH = get_config_value('LOCAL_MODEL_PATH', required=True)

TASKS = [
    'extract_features',
    'train_classifier',
    'classify_features',
    'classify_image'
]

FEATURE_EXTRACTOR_NAMES = [
    'dummy',
    'vgg16_coralnet_ver1',
    'efficientnet_b0_ver1'
]

MODEL_WEIGHTS_SHA = {
    'vgg16':
        'fb83781de0e207ded23bd42d7eb6e75c1e915a6fbef74120f72732984e227cca',
    'efficientnet-b0':
        'c3dc6d304179c6729c0a0b3d4e60c728bdcf0d82687deeba54af71827467204c',
}

CLASSIFIER_TYPES = [
    'LR',
    'MLP'
]

TRAINER_NAMES = [
    'minibatch'
]

MODELS_BUCKET = 'spacer-tools'
TEST_BUCKET = 'spacer-test'

STORAGE_TYPES = [
    's3',
    'filesystem',
    'memory',
    'url'
]

MAX_IMAGE_PIXELS = 10000 * 10000  # 100 mega pixels is max we allow.
MAX_POINTS_PER_IMAGE = 1000

LOCAL_FIXTURE_DIR = str(APP_DIR / 'tests' / 'fixtures')

# The train_classifier task require as least this many images.
MIN_TRAINIMAGES = 10

# Check access to select which tests to run.
HAS_CAFFE = importlib.util.find_spec("caffe") is not None


try:
    s3 = get_s3_conn()
    s3.meta.client.head_bucket(Bucket=TEST_BUCKET)
    HAS_S3_TEST_ACCESS = True
except (botocore.exceptions.ClientError,
        botocore.exceptions.NoCredentialsError):  # pragma: no cover
    logging.info("No connection to spacer-test bucket, "
                 "can't run remote tests")
    HAS_S3_TEST_ACCESS = False

try:
    s3 = get_s3_conn()
    s3.meta.client.head_bucket(Bucket=MODELS_BUCKET)
    HAS_S3_MODEL_ACCESS = True
except (botocore.exceptions.ClientError,
        botocore.exceptions.NoCredentialsError):  # pragma: no cover
    logging.info("No connection to spacer-tools bucket, "
                 "can't run remote tests")
    HAS_S3_MODEL_ACCESS = False

# Add margin to avoid warnings when running unit-test.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS + 20000

# Configure Pillow to be tolerant of image files that are truncated (missing
# data from the last block).
# https://stackoverflow.com/a/23575424/
ImageFile.LOAD_TRUNCATED_IMAGES = True
