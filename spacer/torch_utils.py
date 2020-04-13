"""
This file contains a set of pytorch utility functions
"""

import PIL
import torch
import numpy as np
from torchvision import transforms


def transformation():
    """
    Transform an image and return as a tensor
    :return: a transformer which takes in a image and return a normalized tensor
    """

    transformer = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transformer


def crop_patch(im, crop_size, scale, point_anns, offsets=None):
    """
    Crop patches from an image
    :param im: image for cropping
    :param crop_size: patch size
    :param scale: only supports scale = 1
    :param point_anns: [(row1, col1, 1), (row2, col2, 1), ...]
    :param offsets: offset to the original (row, col)
    :return: patch list and label list
    """

    assert scale == 1, "Only supports scale == 1"

    if offsets is None:
        offsets = np.zeros([len(point_anns), 2])

    patchlist = []
    labellist = []
    pad = crop_size

    im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    for ((row, col, label), offset) in zip(point_anns, offsets):
        center_org = np.asarray([row, col])
        center = np.round(pad + (center_org * scale) + offset).astype(np.int)

        patch = crop_simple(im, center, crop_size)

        patchlist.append(patch)
        labellist.append(label)

    return patchlist, labellist


def crop_simple(im, center, crop_size):
    """ Crops an image around the given center. """
    upper = int(center[0] - crop_size / 2)
    left = int(center[1] - crop_size / 2)
    return im[upper: upper + crop_size, left: left + crop_size, :]


def classify_from_patchlist():
    pass
