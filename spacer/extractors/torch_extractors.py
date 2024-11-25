"""
This file contains a set of pytorch utility functions
"""

from __future__ import annotations
from collections import OrderedDict
from io import BytesIO
from typing import Any

import numpy as np
import torch
from torchvision import transforms

from spacer import config
from spacer import extractors


def transformation():
    """
    Transform an image or numpy array and normalize to [0, 1]
    :return: transformer which takes in a image and return a normalized tensor
    """

    transformer = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transformer


def load_weights(model: Any,
                 weights_datastream: BytesIO) -> Any:
    """
    Load model weights, original weight saved with DataParallel
    Create new OrderedDict that does not contain `module`.
    :param model: Currently support EfficientNet
    :param weights_datastream: model weights, already loaded from storage
    :return: well trained model
    """
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load weights
    state_dicts = torch.load(weights_datastream,
                             map_location=device)

    with config.log_entry_and_exit('model initialization'):
        new_state_dicts = OrderedDict()
        for k, v in state_dicts['net'].items():
            name = k[7:]
            new_state_dicts[name] = v
        model.load_state_dict(new_state_dicts)

    for param in model.parameters():
        param.requires_grad = False
    return model


def extract_feature(patch_list: list,
                    pyparams: dict) -> list:
    """
    Crop patches and extract features
    :param patch_list: a list of cropped images
    :param pyparams: parameter dict
    :return: a list of features
    """
    # Model setup and load pretrained weight
    net = extractors.get_model(model_type=pyparams['model_type'],
                               model_name=pyparams['model_name'],
                               num_classes=pyparams['num_class'])
    net = load_weights(net, pyparams['weights_datastream'])
    net.eval()

    transformer = transformation()

    # Feed forward and extract features
    bs = pyparams['batch_size']
    num_batch = int(np.ceil(len(patch_list) / bs))
    feats_list = []
    with config.log_entry_and_exit('forward pass through net'):
        for b in range(num_batch):
            batch = patch_list[b*bs: b*bs + min(len(patch_list[b*bs:]), bs)]
            batch = torch.stack([transformer(i) for i in batch])
            with torch.no_grad():
                features = net.extract_features(batch)
            feats_list.extend(features.tolist())

    return feats_list
