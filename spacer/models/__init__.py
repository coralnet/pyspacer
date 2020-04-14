"""
For possible future network extension
"""
from .efficientnet import EfficientNet


def get_model(model_type, model_name, num_classes=1000):
    if model_type == 'efficientnet':
        model = EfficientNet.from_pretrained(model_name, num_classes)
    else:
        raise NotImplementedError(model_name)
    return model
