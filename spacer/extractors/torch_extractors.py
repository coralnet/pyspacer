"""
This file contains a set of pytorch utility functions
"""

from __future__ import annotations
import abc
from collections import OrderedDict
from io import BytesIO

import numpy as np
import torch
from torchvision import transforms

from spacer import config
from spacer.messages import DataLocation
from spacer.storage import storage_factory
from .base import FeatureExtractor


def transformation():
    """
    Transform an image or numpy array and normalize to [0, 1]
    :return: transformer which takes in a image and return a normalized tensor
    """

    transformer = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transformer


class TorchExtractor(FeatureExtractor, abc.ABC):

    # weights should be a PyTorch tensor file, typically .pt
    DATA_LOCATION_KEYS = ['weights']
    BATCH_SIZE = 10

    def patches_to_features(self, patch_list):

        # Load pretrained weights
        weights_datastream, extractor_loaded_remotely = (
            self.load_datastream('weights'))
        net = self.load_weights(weights_datastream)
        net.eval()

        transformer = transformation()

        # Feed forward and extract features
        batch_size = self.BATCH_SIZE
        num_batches = int(np.ceil(len(patch_list) / batch_size))
        feats_list = []
        with config.log_entry_and_exit('forward pass through net'):
            for b in range(num_batches):
                this_batch_size = min(
                    len(patch_list[b*batch_size:]), batch_size)
                batch = patch_list[
                    b*batch_size:(b*batch_size + this_batch_size)]
                batch = torch.stack([transformer(i) for i in batch])
                with torch.no_grad():
                    features = net.extract_features(batch)
                feats_list.extend(features.tolist())

        return features, extractor_loaded_remotely

    @staticmethod
    def untrained_model() -> torch.nn.Module:
        """
        Return a model that's been initialized with the intended
        parameters, but not trained at all yet.
        """
        raise NotImplementedError

    @classmethod
    def load_weights(cls,
                     weights_datastream: BytesIO) -> torch.nn.Module:
        """
        Load model weights, which may have been saved with DataParallel
        (as CoralNet 1.0's weights were).
        Create new OrderedDict that does not contain `module`.
        :param weights_datastream: model weights, already loaded from storage
        :return: well trained model
        """
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = cls.untrained_model()

        # Load weights
        state_dicts = torch.load(weights_datastream,
                                 map_location=device)

        with config.log_entry_and_exit('model initialization'):
            new_state_dicts = OrderedDict()
            for k, v in state_dicts['net'].items():
                # Instead of e.g. `module._conv_stem.weight`,
                # strip that first part to get `_conv_stem.weight`.
                # The `module.` prefix comes from saving the weights with
                # DataParallel.
                # https://github.com/pytorch/pytorch/issues/9176
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dicts[k] = v
            model.load_state_dict(new_state_dicts)

        for param in model.parameters():
            param.requires_grad = False
        return model

    @classmethod
    def untrained_instance(cls, data_loc_key='weights'):
        """
        This is an EfficientNetExtractor that we can use without fixtures,
        when we only care about valid format and not 'legit'
        features / predictions (no training epochs are actually run on the
        extractor).
        """
        model = cls.untrained_model()
        state = dict(net=model.state_dict())
        with BytesIO() as stream:
            torch.save(state, stream)
            storage_factory('memory').store(data_loc_key, stream)
        weights_loc = DataLocation('memory', data_loc_key)
        return cls(data_locations=dict(weights=weights_loc))
