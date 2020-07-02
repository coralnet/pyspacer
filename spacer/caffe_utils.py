"""
This file contains a set of caffe utility functions copied into this repo for
simplicity. Since support for Caffe will be deprecate,
these are only lightly cleaned up from their original state.
"""

import logging
from copy import copy
from functools import lru_cache
from typing import List, Any, Tuple

import caffe
import time
import hashlib
import numpy as np

from spacer import config


class Transformer:
    """
    Transformer is a class for preprocessing and deprocessing images
    according to the vgg16 pre-processing paradigm.
    (scaling and mean subtraction.).
    """

    def __init__(self, mean: Tuple = (0, 0, 0)) -> None:
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


def classify_from_imlist(im_list: List,
                         net: Any,
                         transformer: Transformer,
                         batch_size: int,
                         scorelayer: str = 'score',
                         startlayer: str = 'conv1_1') -> List:
    """
    classify_from_imlist classifies a list of images and returns
    estimated labels and scores.
    Only support classification nets (not FCNs).
    :param im_list: list of images to classify (each stored as a numpy array).
    :param net: caffe net object.
    :param transformer: transformer object as defined above.
    :param batch_size: batch size for the net.
    :param scorelayer: name of the score (the last conv) layer.
    :param startlayer: name of first convolutional layer.
    :return: features list.
    """

    scorelist = []
    for b in range(len(im_list) // batch_size + 1):
        for i in range(batch_size):
            pos = b * batch_size + i
            if pos < len(im_list):
                net.blobs['data'].data[i, :, :, :] = \
                    transformer.preprocess(im_list[pos])
        net.forward(start=startlayer)
        scorelist.extend(list(copy(net.blobs[scorelayer].data).
                              astype(np.float)))

    scorelist = scorelist[:len(im_list)]

    return scorelist


@lru_cache(maxsize=1)
def load_net(modeldef_path: str,
             modelweighs_path: str) -> Any:
    """
    load pretrained net.
    :param modeldef_path: model path.
    :param modelweighs_path: pretrained weights path.
    :return: pretrained model.
    """
    # To verify that the correct weight is loaded
    start = time.time()
    with open(modelweighs_path, 'rb') as fp:
        sha256 = hashlib.sha256(fp.read()).hexdigest()
    assert sha256 == config.MODEL_WEIGHTS_SHA['vgg16']
    logging.debug("-> Time spent on checking SHA: {}".format(
        time.time() - start
    ))
    return caffe.Net(modeldef_path, modelweighs_path, caffe.TEST)


def classify_from_patchlist(patchlist: List,
                            pyparams: dict,
                            modeldef_path: str,
                            modelweighs_path: str,
                            scorelayer: str = 'score',
                            startlayer: str = 'conv1_1') -> List:
    """
    extract features of a list of patches
    :param patchlist: a list of patches (cropped images).
    :param pyparams: a set of parameters.
    :param modeldef_path: model path.
    :param modelweighs_path: pretrained weights path.
    :param scorelayer: name of the score (the last conv) layer.
    :param startlayer: name of first convolutional layer.
    :return: a list of features
    """
    # Setup caffe
    caffe.set_mode_cpu()
    net = load_net(modeldef_path, modelweighs_path)

    # Classify
    transformer = Transformer(pyparams['im_mean'])
    scorelist = classify_from_imlist(
        patchlist, net, transformer, pyparams['batch_size'],
        scorelayer=scorelayer, startlayer=startlayer
    )

    return scorelist
