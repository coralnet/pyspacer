"""
Contains config and settings for the repo.
"""

import importlib
import json
import logging
import os
import sys
import time
import warnings
from contextlib import ContextDecorator
from pathlib import Path
from typing import Any, Optional

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


# One way to specify settings is through a secrets.json file. Example:
# {
#     "AWS_ACCESS_KEY_ID": "...",
#     "AWS_SECRET_ACCESS_KEY": "..."
# }
SECRETS_PATH = REPO_DIR / 'secrets.json'
SECRETS = None
if SECRETS_PATH.exists():
    with open(SECRETS_PATH) as fp:
        SECRETS = json.load(fp)


# Another way is through Django settings. Example:
# SPACER = {
#     'AWS_ACCESS_KEY_ID': '...',
#     'AWS_SECRET_ACCESS_KEY': '...',
# }
SETTINGS_FROM_DJANGO: Optional[dict] = None
try:
    from django.core.exceptions import ImproperlyConfigured
except ImportError:
    pass
else:
    # This by itself shouldn't get errors.
    from django.conf import settings

    # If settings module can't be found, this gets ImproperlyConfigured.
    # If the module can be found, but the SPACER setting is absent, this
    # gets AttributeError.
    try:
        SETTINGS_FROM_DJANGO = settings.SPACER
    except (ImproperlyConfigured, AttributeError):
        pass


def get_config_detection_result():
    result = ""

    if SECRETS:
        result += "Secrets file found."
    else:
        result += "Secrets file not found."

    if SETTINGS_FROM_DJANGO:
        result += " SPACER Django setting found."
    else:
        result += " SPACER Django setting not found."

    return result


def get_config_value(key: str, default: Any = 'NO_DEFAULT') -> Any:

    def is_valid_value(value_):
        # Treat an empty string the same as not specifying a setting.
        return value_ not in ['', None]

    # Try environment variables. Each should be prefixed with 'SPACER_'.
    value = os.getenv('SPACER_' + key)
    if is_valid_value(value):
        return value

    def handle_unspecified_setting():
        if default == 'NO_DEFAULT':
            raise RuntimeError(
                f"{key} setting is required."
                f" (Debug info: {get_config_detection_result()})"
            )
        return default

    # Try secrets file.
    if SECRETS:
        value = SECRETS.get(key)
        if is_valid_value(value):
            return value
        return handle_unspecified_setting()

    # Try Django settings.
    if SETTINGS_FROM_DJANGO:
        try:
            value = SETTINGS_FROM_DJANGO[key]
        except KeyError:
            return handle_unspecified_setting()
        else:
            if is_valid_value(value):
                return value
            return handle_unspecified_setting()

    return handle_unspecified_setting()


def get_s3_conn():
    """
    Returns a boto s3 connection.
    - It first looks for credentials in spacer config.
    - If not found there it will default to credentials in ~/.aws/credentials
    """
    return boto3.resource('s3',
                          region_name="us-west-2",
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY)


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


AWS_ACCESS_KEY_ID = get_config_value('AWS_ACCESS_KEY_ID', default=None)
AWS_SECRET_ACCESS_KEY = get_config_value('AWS_SECRET_ACCESS_KEY', default=None)

LOCAL_MODEL_PATH = get_config_value('LOCAL_MODEL_PATH')

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

MAX_IMAGE_PIXELS = get_config_value('MAX_IMAGE_PIXELS', default=10000*10000)
MAX_POINTS_PER_IMAGE = get_config_value('MAX_POINTS_PER_IMAGE', default=1000)

LOCAL_FIXTURE_DIR = str(APP_DIR / 'tests' / 'fixtures')

# The train_classifier task requires as least this many images.
MIN_TRAINIMAGES = get_config_value('MIN_TRAINIMAGES', default=10)

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


CONFIGURABLE_VARS = [
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'LOCAL_MODEL_PATH',
    'MAX_IMAGE_PIXELS',
    'MAX_POINTS_PER_IMAGE',
    'MIN_TRAINIMAGES',
]


def check():
    """
    Print values of all configurable variables.
    """
    print(get_config_detection_result())

    for var_name in CONFIGURABLE_VARS:
        # Get the var_name attribute in the current module
        var_value = getattr(sys.modules[__name__], var_name)

        if '_KEY' in var_name:
            # Treat this as a sensitive value; don't print the entire thing
            value_display = f'{var_value[:6]} ... {var_value[-6:]}'
        else:
            value_display = str(var_value)

        print(f"{var_name}: {value_display}")
