"""
Contains config and settings for the repo.
"""

from __future__ import annotations
import importlib
import json
import os
import sys
import threading
import time
import warnings
from contextlib import ContextDecorator
from logging import basicConfig, getLogger
from pathlib import Path
from typing import Any

import boto3
from PIL import Image, ImageFile

from spacer.exceptions import ConfigError

logger = getLogger(__name__)


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
SETTINGS_FROM_DJANGO: dict | None = None
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
            raise ConfigError(
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


# If your end application/script doesn't configure its own logging,
# or if you're running unit tests and want log output, you can specify this
# config value to output logs to console or to a file of your choice.
# Specify either as 'console', or as an absolute path to the desired file.
LOG_DESTINATION = get_config_value('LOG_DESTINATION', default=None)
# And this is the log level to use when logging to that destination.
# Specify as "INFO", etc.
LOG_LEVEL = get_config_value('LOG_LEVEL', default='INFO')

if LOG_DESTINATION:
    if LOG_DESTINATION == 'console':
        filename = None
    else:
        filename = LOG_DESTINATION

    basicConfig(
        level=LOG_LEVEL,
        filename=filename,
        format='%(asctime)s - %(levelname)s:%(name)s\n%(message)s',
    )


# Save S3 connections (resources) for reuse, but only have one per thread,
# because they're not thread-safe:
# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/resources.html
THREAD_LOCAL = threading.local()


def get_s3_conn():
    """
    Returns a boto s3 connection.
    Each thread only establishes a connection once, saving it to
    THREAD_LOCAL.s3_connection and reusing it thereafter.
    """
    if not all([AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        raise ConfigError(
            "All AWS config variables must be specified to use S3.")

    try:
        THREAD_LOCAL.s3_connection
    except AttributeError:
        # This passes credentials from spacer config. If credentials are
        # None, it will default to using credentials in ~/.aws/credentials
        THREAD_LOCAL.s3_connection = boto3.resource(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

        # We're interested in confirming:
        # - That this only gets reached once per thread (log with %(thread)d
        #   in the logging format to confirm this)
        # - How long a single resource retrieval can be reused before
        #   expiring (if it ever expires)
        logger.info("Called boto3.resource() in get_s3_conn()")

    return THREAD_LOCAL.s3_connection


class log_entry_and_exit(ContextDecorator):
    def __init__(self, name):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.debug('Entering: %s', self.name)

    def __exit__(self, exc_type, exc, exc_tb):
        logger.debug('Exiting: %s after %f seconds.', self.name,
                     time.time() - self.start_time)


AWS_ACCESS_KEY_ID = get_config_value('AWS_ACCESS_KEY_ID', default=None)
AWS_SECRET_ACCESS_KEY = get_config_value('AWS_SECRET_ACCESS_KEY', default=None)
AWS_REGION = get_config_value('AWS_REGION', default=None)

# Filesystem directory to use for caching downloaded feature-extractor data.
# This will be used whenever there is s3 or url based extractor data.
EXTRACTORS_CACHE_DIR = get_config_value('EXTRACTORS_CACHE_DIR', default=None)

TASKS = [
    'extract_features',
    'train_classifier',
    'classify_features',
    'classify_image'
]

CLASSIFIER_TYPES = [
    'LR',
    'MLP'
]

TRAINER_NAMES = [
    'minibatch'
]

# For extractors used in unit tests.
TEST_EXTRACTORS_BUCKET = get_config_value(
    'TEST_EXTRACTORS_BUCKET', default=None)
# For other fixtures used in unit tests.
#
# At least for now, the main reason these bucket names are pulled from
# config is to not expose the bucket names used by the PySpacer core devs.
# However, since these test files are not publicly linked and need to
# live in an S3 bucket with specific filenames (specified by TEST_EXTRACTORS
# and individual tests), the tests are still onerous to set up for anyone
# besides the core devs. This should be addressed sometime.
TEST_BUCKET = get_config_value('TEST_BUCKET', default=None)
# A few other fixtures live here.
LOCAL_FIXTURE_DIR = str(APP_DIR / 'tests' / 'fixtures')

STORAGE_TYPES = [
    's3',
    'filesystem',
    'memory',
    'url'
]

MAX_IMAGE_PIXELS = get_config_value('MAX_IMAGE_PIXELS', default=10000*10000)
MAX_POINTS_PER_IMAGE = get_config_value('MAX_POINTS_PER_IMAGE', default=1000)

# The train_classifier task requires as least this many images.
MIN_TRAINIMAGES = get_config_value('MIN_TRAINIMAGES', default=10)

# Check access to select which tests to run.
HAS_CAFFE = importlib.util.find_spec("caffe") is not None


# Add margin to avoid warnings when running unit-test.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS + 20000

# Configure Pillow to be tolerant of image files that are truncated (missing
# data from the last block).
# https://stackoverflow.com/a/23575424/
ImageFile.LOAD_TRUNCATED_IMAGES = True


CONFIGURABLE_VARS = [
    # These two variables are required if you're using AWS S3 storage,
    # unless spacer is running on an AWS instance which has been set up with
    # `aws configure`.
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    # This is required if you're using S3 storage.
    'AWS_REGION',
    # This is required if you're loading feature extractors from a remote
    # source (S3 or URL).
    'EXTRACTORS_CACHE_DIR',
    # These are required to run certain unit tests. They're also not really
    # usable by anyone besides spacer's core devs at the moment.
    'TEST_EXTRACTORS_BUCKET',
    'TEST_BUCKET',
    # These can just be configured as needed, or left as defaults.
    'LOG_DESTINATION',
    'LOG_LEVEL',
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

        if var_value and '_KEY' in var_name:
            # Treat this as a sensitive value; don't print the entire thing
            value_display = f'{var_value[:6]} ... {var_value[-6:]}'
        else:
            value_display = str(var_value)

        print(f"{var_name}: {value_display}")
