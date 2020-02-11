import importlib
import warnings

import boto

warnings.simplefilter("ignore", ResourceWarning)

HAS_CAFFE = importlib.util.find_spec("caffe") is not None
LOCAL_MODEL_PATH = '/workspace/models'

FEATURE_EXTRACTOR_NAMES = ['dummy', 'vgg16_coralnet_ver1', 'efficientnet_b0_imagenet']

MODELS_BUCKET = 'spacer-tools'

STORAGE_TYPES = ['s3', 'local']

try:
    conn = boto.connect_s3()
    bucket = conn.get_bucket('spacer-test', validate=True)
    HAS_S3_TEST_ACCESS = True
except _:
    HAS_S3_TEST_ACCESS = False
