"""
Contains config and settings for the repo.
"""

from __future__ import annotations
import importlib
import json
import os
import sys
import time
import warnings
from contextlib import ContextDecorator
from logging import basicConfig, getLogger
from pathlib import Path
from typing import Any

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


def get_config_value(
        key: str,
        value_type: type = str,
        default: Any = 'NO_DEFAULT') -> Any:

    def is_valid_value(value_):
        # Treat an empty string the same as not specifying a setting.
        return value_ not in ['', None]

    def cast_str_value(value_):
        if value_type == str:
            return value_

        if value_type == int:
            return int(value_)

        if value_type == bool:
            match value_:
                case "True":
                    return True
                case "False":
                    return False
                case _:
                    raise ConfigError(f"{key} may only be True or False.")

        assert False, "Don't call cast_str_value() with unsupported types."

    # Try environment variables. Each should be prefixed with 'SPACER_'.
    value = os.getenv('SPACER_' + key)
    if is_valid_value(value):
        return cast_str_value(value)

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
            return cast_str_value(value)
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
AWS_SESSION_TOKEN = get_config_value('AWS_SESSION_TOKEN', default=None)
AWS_PROFILE_NAME = get_config_value('AWS_PROFILE_NAME', default=None)

# If True, AWS is accessed without any credentials, which can simplify setup
# while still allowing access to public S3 files.
AWS_ANONYMOUS = get_config_value(
    'AWS_ANONYMOUS', value_type=bool, default=False)

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

# Amazon S3 bucket for temporarily storing data during unit tests.
# You'll need write access to this bucket to run the applicable tests.
TEST_BUCKET = get_config_value('TEST_BUCKET', default=None)
# A few testing fixtures live here.
LOCAL_FIXTURE_DIR = str(APP_DIR / 'tests' / 'fixtures')

# And the rest of the testing fixtures live in these CoralNet-owned
# private buckets. (CoralNet devs should specify the names of the buckets
# in their environment.)
# These tests and fixtures should be reorganized sometime so that anyone can
# run the applicable tests.
CN_TEST_EXTRACTORS_BUCKET = get_config_value(
    'CN_TEST_EXTRACTORS_BUCKET', default=None)
CN_FIXTURES_BUCKET = get_config_value('CN_FIXTURES_BUCKET', default=None)

STORAGE_TYPES = [
    's3',
    'filesystem',
    'memory',
    'url'
]

MAX_IMAGE_PIXELS = get_config_value(
    'MAX_IMAGE_PIXELS', value_type=int, default=10000*10000)
MAX_POINTS_PER_IMAGE = get_config_value(
    'MAX_POINTS_PER_IMAGE', value_type=int, default=1000)

# Size of training batches. This number of features must be able to fit
# in memory. Raising this allows the reference set to be larger,
# which can improve calibration results.
TRAINING_BATCH_LABEL_COUNT = get_config_value(
    'TRAINING_BATCH_LABEL_COUNT', value_type=int, default=5000)

# Check access to select which tests to run.
HAS_CAFFE = importlib.util.find_spec("caffe") is not None


# Add margin to avoid warnings when running unit-test.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS + 20000

# Configure Pillow to be tolerant of image files that are truncated (missing
# data from the last block).
# https://stackoverflow.com/a/23575424/
ImageFile.LOAD_TRUNCATED_IMAGES = True


CONFIGURABLE_VARS = [
    # These variables enable a few ways to provide credentials for AWS S3
    # storage.
    # None of these are required for:
    # - Workflows that don't use S3.
    # - Certain S3 credentials methods, such as using the AWS instance
    #   metadata service without a profile name.
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'AWS_SESSION_TOKEN',
    'AWS_PROFILE_NAME',
    'AWS_ANONYMOUS',
    'AWS_REGION',
    # This is required if you're loading feature extractors from a remote
    # source (S3 or URL).
    'EXTRACTORS_CACHE_DIR',
    # This is required for S3 unit tests.
    'TEST_BUCKET',
    # These are required to run certain unit tests. They're also only usable
    # by CoralNet devs at the moment.
    'CN_TEST_EXTRACTORS_BUCKET',
    'CN_FIXTURES_BUCKET',
    # These can just be configured as needed, or left as defaults.
    'LOG_DESTINATION',
    'LOG_LEVEL',
    'MAX_IMAGE_PIXELS',
    'MAX_POINTS_PER_IMAGE',
    'TRAINING_BATCH_LABEL_COUNT',
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
