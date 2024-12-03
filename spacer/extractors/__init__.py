"""
For possible future network extension
"""
from .base import DummyExtractor, FeatureExtractor
from .efficientnet import EfficientNetExtractor
from .vgg16 import VGG16CaffeExtractor


__all__ = [
    'DummyExtractor',
    'EfficientNetExtractor',
    'FeatureExtractor',
    'VGG16CaffeExtractor',
]
