"""
This file contains a set of caffe utility functions copied into this repo for
simplicity. Since support for Caffe will be deprecate,
these are only lightly cleaned up from their original state.
"""

from copy import copy
from functools import lru_cache
from typing import List, Tuple

import caffe
import numpy as np
from PIL import Image


class Transformer:
    """
    Transformer is a class for preprocessing and deprocessing images
    according to the vgg16 pre-processing paradigm
    (scaling and mean subtraction.)
    """

    def __init__(self, mean=(0, 0, 0)):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0

    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occurring
        in the vgg16 caffe prototxt.
        """

        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        im *= self.scale
        im = im.transpose((2, 0, 1))

        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)


def classify_from_imlist(im_list, net, transformer, batch_size,
                         scorelayer='score', startlayer='conv1_1'):
    """
    classify_from_imlist classifies a list of images and returns
    estimated labels and scores.
    Only support classification nets (not FCNs).

    Takes
    im_list: list of images to classify (each stored as a numpy array).
    net: caffe net object
    transformer: transformer object as defined above.
    batch_size: batch size for the net.
    scorelayer: name of the score layer.
    startlayer: name of first convolutional layer.
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
def load_net(modeldef_path, modelweighs_path):
    return caffe.Net(modeldef_path, modelweighs_path, caffe.TEST)


def classify_from_patchlist(patchlist: List,
                            pyparams: dict,
                            modeldef_path: str,
                            modelweighs_path: str,
                            scorelayer: str = 'score',
                            startlayer: str = 'conv1_1'):
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
