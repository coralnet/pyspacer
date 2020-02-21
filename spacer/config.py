import importlib
import os
import warnings

import boto

# Per discussion in https://github.com/boto/boto3/issues/454,
# the boto package is raising a lot of warnings that it shouldn't.
warnings.simplefilter("ignore", ResourceWarning)


LOCAL_MODEL_PATH = '/workspace/models'

FEATURE_EXTRACTOR_NAMES = ['dummy',
                           'vgg16_coralnet_ver1',
                           'efficientnet_b0_imagenet']

MODELS_BUCKET = 'spacer-tools'

STORAGE_TYPES = ['s3', 'local', 'memory']

LOCAL_FIXTURE_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'tests', 'fixtures')

# Check access to select which tests to run.
HAS_CAFFE = importlib.util.find_spec("caffe") is not None

try:
    conn = boto.connect_s3()
    bucket = conn.get_bucket('spacer-test', validate=True)
    HAS_S3_TEST_ACCESS = True
except _:
    HAS_S3_TEST_ACCESS = False

try:
    conn = boto.connect_s3()
    bucket = conn.get_bucket('spacer-tools', validate=True)
    HAS_S3_MODEL_ACCESS = True
except _:
    HAS_S3_MODEL_ACCESS = False
