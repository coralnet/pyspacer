"""
This file contains a set of caffe utiiity functions copied into this repo for simplicity.
Since support for Caffe will be deprecate, these are not cleaned up from their original state.
"""

import os

import numpy as np

from copy import copy

from PIL import Image


class Transformer:
    """
    Transformer is a class for preprocessing and deprocessing images
    according to the vgg16 pre-processing paradigm (scaling and mean subtraction.)
    """

    def __init__(self, mean=(0, 0, 0)):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occuring in the vgg16 caffe prototxt.
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


def classify_from_imlist(im_list, net, transformer, batch_size, scorelayer='score', startlayer='conv1_1'):
    """
    classify_from_imlist classifies a list of images and returns estimated labels and scores.
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
                net.blobs['data'].data[i, :, :, :] = transformer.preprocess(im_list[pos])
        net.forward(start=startlayer)
        scorelist.extend(list(copy(net.blobs[scorelayer].data).astype(np.float)))

    scorelist = scorelist[:len(im_list)]
    estlist = [np.argmax(s) for s in scorelist]

    return estlist, scorelist


def rgb2gray(im):
    """ Converts RGB to gray scale """
    im = copy(im)
    m = np.mean(im, axis=2)
    for i in range(3):
        im[:, :, i] = m
    return im


def gray2rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret


def crop_patch(im, crop_size, scale, point_anns, offsets=None):
    """ Crops patches from images. """

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


def classify_from_patchlist(imdict,
                            pyparams,
                            modeldef_path,
                            modelweighs_path,
                            storage,
                            scorelayer='score',
                            startlayer='conv1_1'):
    # Setup caffe
    caffe.set_mode_cpu()
    net = caffe.Net(modeldef_path, modelweighs_path, caffe.TEST)

    scale = 1
    estlist, scorelist, gtlist = [], [], []
    transformer = Transformer(pyparams['im_mean'])
    for imname in imdict:

        (point_anns, height_cm) = imdict[os.path.basename(imname)]

        # Load image
        im_pil = storage.load_image(imname)
        im = np.asarray(im_pil)

        # Do it a second time due to bug
        im = np.asarray(im_pil)
        if len(im.shape) == 2 or im.shape[2] == 1:
            im = gray2rgb(im)
        im = im[:, :, :3]  # only keep the first three color channels

        # Crop patches
        patchlist, this_gtlist = crop_patch(im, pyparams['crop_size'], scale, point_anns)

        # Classify
        [this_estlist, this_scorelist] = classify_from_imlist(patchlist, net, transformer, pyparams['batch_size'],
                                                              scorelayer=scorelayer, startlayer=startlayer)

        # Store
        gtlist.extend(this_gtlist)
        estlist.extend(this_estlist)
        scorelist.extend(this_scorelist)

    return gtlist, estlist, scorelist