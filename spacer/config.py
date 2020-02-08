import importlib

HAS_CAFFE = importlib.util.find_spec("caffe") is not None
LOCAL_MODEL_PATH = '/workspace/models'

FEATURE_EXTRACTOR_NAMES = ['vgg16_coralnet_ver1', 'efficientnet_b0_imagenet']

MODELS_BUCKET = 'spacer-tools'

STORAGE_TYPES = ['s3', 'local']
