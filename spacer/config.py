import importlib
import json
import os
import warnings
import boto

from boto import sqs

# Per discussion in https://github.com/boto/boto3/issues/454,
# the boto package is raising a lot of warnings that it shouldn't.
warnings.simplefilter("ignore", ResourceWarning)


# Load secrets from secrets.json
secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'secrets.json')

if not os.path.exists(secrets_path):  # pragma: no cover
    RuntimeWarning('secrets.json not found')
    AWS_ACCESS_KEY_ID = ''
    AWS_SECRET_ACCESS_KEY = ''
else:
    try:
        with open(secrets_path) as fp:
            secrets = json.load(fp)
        AWS_ACCESS_KEY_ID = secrets['AWS_ACCESS_KEY_ID']
        AWS_SECRET_ACCESS_KEY = secrets['AWS_SECRET_ACCESS_KEY']
    except Exception as err:  # pragma: no cover
        RuntimeWarning('Unable to parse secrets.json: {}'.format(repr(err)))
        AWS_ACCESS_KEY_ID = ''
        AWS_SECRET_ACCESS_KEY = ''


LOCAL_MODEL_PATH = '/workspace/models'

TASKS = [
    'extract_features',
    'train_classifier',
    'deploy'
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
    'memory'
]

LOCAL_FIXTURE_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'tests', 'fixtures')

# The train_classifier task require as least this many images.
MIN_TRAINIMAGES = 10

# Check access to select which tests to run.
HAS_CAFFE = importlib.util.find_spec("caffe") is not None

conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
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

inqueue = sqs.connect_to_region(
        "us-west-2",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
).get_queue('spacer_test_jobs')

if inqueue is None:  # pragma: no cover
    HAS_SQS_QUEUE_ACCESS = False
    print('-> No access to SQS found.')
else:
    HAS_SQS_QUEUE_ACCESS = True
