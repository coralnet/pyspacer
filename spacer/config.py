"""
Contains config and settings for the repo.
"""

import importlib
import json
import os
import warnings
from typing import Tuple, Optional

import boto
from boto import sqs

# Per discussion in https://github.com/boto/boto3/issues/454,
# the boto package is raising a lot of warnings that it shouldn't.
warnings.simplefilter("ignore", ResourceWarning)


def get_secret(key):
    """ Try to load settings from secrets.json file """
    secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'secrets.json')
    if not os.path.exists(secrets_path):  # pragma: no cover
        RuntimeWarning('secrets.json not found')
        return None
    else:
        try:
            with open(secrets_path) as fp:
                secrets = json.load(fp)
            return secrets[key]
        except Exception as err:  # pragma: no cover
            RuntimeWarning(
                'Unable to parse secrets.json: {}'.format(repr(err)))
            return None


def get_aws_credentials() -> Tuple[Optional[str], Optional[str]]:
    aws_key_id = os.getenv('SPACER_AWS_ACCESS_KEY_ID')
    aws_key_secret = os.getenv('SPACER_AWS_SECRET_ACCESS_KEY')

    if not aws_key_id:
        aws_key_id = get_secret('SPACER_AWS_ACCESS_KEY_ID')
    if not aws_key_secret:
        aws_key_secret = get_secret('SPACER_AWS_SECRET_ACCESS_KEY')

    return aws_key_id, aws_key_secret


def get_s3_conn():
    """
    Returns a boto s3 connection.
    - It first looks for credentials in the environmental vars.
    - If not found there it looks in secrets.json
    - If not found there it will default to credentials in ~/.aws/credentials
    """
    aws_key_id, aws_key_secret = get_aws_credentials()
    return boto.connect_s3(aws_key_id, aws_key_secret)


def get_sqs_conn():
    """
    Returns a connection to SQS.
    - It first looks for credentials in the environmental vars.
    - If not found there it looks in secrets.json
    - If not found there it will default to credentials in ~/.aws/credentials
    """
    aws_key_id, aws_key_secret = get_aws_credentials()
    return sqs.connect_to_region(
        "us-west-2",
        aws_access_key_id=aws_key_id,
        aws_secret_access_key=aws_key_secret)


def get_local_model_path():
    local_model_path = os.getenv('SPACER_LOCAL_MODEL_PATH')
    if local_model_path is None:
        return get_secret('SPACER_LOCAL_MODEL_PATH')
    return local_model_path


LOCAL_MODEL_PATH = get_local_model_path()

assert LOCAL_MODEL_PATH is not None, \
    "SPACER_LOCAL_MODEL_PATH environmental variable must be set."

assert os.path.exists(LOCAL_MODEL_PATH), "LOCAL_MODEL_PATH is set, " \
                                         "but path doesn't exist"

TASKS = [
    'extract_features',
    'train_classifier',
    'classify_features',
    'classify_image'
]

FEATURE_EXTRACTOR_NAMES = [
    'dummy',
    'vgg16_coralnet_ver1',
    'efficientnet_b0_imagenet'
]

TRAINER_NAMES = [
    'dummy',
    'minibatch'
]

MODELS_BUCKET = 'spacer-tools'

STORAGE_TYPES = [
    's3',
    'filesystem',
    'memory',
    'url'
]

LOCAL_FIXTURE_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'tests', 'fixtures')

# The train_classifier task require as least this many images.
MIN_TRAINIMAGES = 10

# Check access to select which tests to run.
HAS_CAFFE = importlib.util.find_spec("caffe") is not None

conn = get_s3_conn()
try:
    bucket = conn.get_bucket('spacer-test', validate=True)
    HAS_S3_TEST_ACCESS = True
except boto.exception.S3ResponseError as err:  # pragma: no cover
    print("-> No connection to spacer-test bucket, can't run remote tests")
    HAS_S3_TEST_ACCESS = False

try:
    bucket = conn.get_bucket('spacer-tools', validate=True)
    HAS_S3_MODEL_ACCESS = True
except boto.exception.S3ResponseError as err:  # pragma: no cover
    print("-> No connection to spacer-tools bucket, can't download models")
    HAS_S3_MODEL_ACCESS = False

inqueue = get_sqs_conn().get_queue('spacer_test_jobs')

if inqueue is None:  # pragma: no cover
    HAS_SQS_QUEUE_ACCESS = False
    print('-> No access to SQS found.')
else:
    HAS_SQS_QUEUE_ACCESS = True


