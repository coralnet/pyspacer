"""
This file contains a set of caffe utility functions copied into this repo for
simplicity. Since support for Caffe will be deprecate,
these are only lightly cleaned up from their original state.
"""

from __future__ import annotations
from copy import copy
from functools import lru_cache
from typing import Any

import numpy as np

from spacer import config
from spacer.exceptions import ConfigError
from .base import FeatureExtractor


class Transformer:
    """
    Transformer is a class for preprocessing and deprocessing images
    according to the vgg16 pre-processing paradigm.
    (scaling and mean subtraction.).
    """

    def __init__(self, mean: tuple = (0, 0, 0)) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0

    def preprocess(self, im: np.ndarray) -> np.ndarray:
        """
        preprocess() emulate the pre-processing occurring
        in the vgg16 caffe prototxt.
        :param im: numpy array.
        :return: normalized numpy array.
        """
        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        im *= self.scale
        im = im.transpose((2, 0, 1))

        return im

    def deprocess(self, im: np.ndarray) -> np.ndarray:
        """
        inverse of preprocess().
        :param im: normalized numpy array.
        :return: original image.
        """
        im = im.transpose((1, 2, 0))
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)


class VGG16CaffeExtractor(FeatureExtractor):

    # definition should be a Caffe prototxt file, typically .prototxt
    # weights should be a Caffe model file, typically .caffemodel
    DATA_LOCATION_KEYS = ['definition', 'weights']

    BATCH_SIZE = 10
    # Name of first convolutional layer.
    START_LAYER = 'conv1_1'
    # Name of the score (the last conv) layer.
    SCORE_LAYER = 'fc7'

    def __call__(self, im, rowcols):
        if not config.HAS_CAFFE:
            raise ConfigError(
                f"Need Caffe installed to call"
                f" {self.__class__.__name__}.")

        return super().__call__(im, rowcols)

    def patches_to_features(self, patch_list):
        # Load pretrained weights
        definition_filepath, _ = (
            self.load_data_into_filesystem('definition'))
        weights_filepath, extractor_loaded_remotely = (
            self.load_data_into_filesystem('weights'))
        net = load_net(definition_filepath, weights_filepath)

        # Extract features.
        # Although the below code is somewhat network-agnostic, it's only
        # meant for classification nets (not FCNs).

        transformer = Transformer((128, 128, 128))

        with config.log_entry_and_exit('forward pass through net'):
            features = []
            for b in range(len(patch_list) // self.BATCH_SIZE + 1):
                for i in range(self.BATCH_SIZE):
                    pos = b * self.BATCH_SIZE + i
                    if pos < len(patch_list):
                        net.blobs['data'].data[i, :, :, :] = \
                            transformer.preprocess(patch_list[pos])
                net.forward(start=self.START_LAYER)
                features.extend(list(
                    copy(net.blobs[self.SCORE_LAYER].data).astype(float)
                ))

            features = features[:len(patch_list)]

        features = [feat.tolist() for feat in features]
        return features, extractor_loaded_remotely

    @property
    def feature_dim(self):
        return 4096


@lru_cache(maxsize=1)
def load_net(modeldef_path: str,
             modelweights_path: str) -> Any:
    """
    Load pretrained net.
    :param modeldef_path: model path.
    :param modelweights_path: pretrained weights path.
    :return: pretrained model.
    """
    # Should have checked for a Caffe installation before reaching this.
    import caffe
    caffe.set_mode_cpu()
    return caffe.Net(modeldef_path, modelweights_path, caffe.TEST)
