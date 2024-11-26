"""
For possible future network extension
"""
from .base import (
    DummyExtractor,
    FeatureExtractor,
    VGG16CaffeExtractor,
)
from .efficientnet import EfficientNetExtractor


__all__ = [
    'DummyExtractor',
    'EfficientNetExtractor',
    'FeatureExtractor',
    'VGG16CaffeExtractor',
]
