import numpy as np


def gray2rgb(im):
    """
    Convert gray image to RGB image
    :param im: gray image to be converted
    :return: RGB image
    """
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im

    return ret


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
    """
    Crops an image around the given center
    :param im: image to be cropped
    :param center: offset (row, col)
    :param crop_size: cropping size
    :return: cropped image in numpy array
    """
    upper = int(center[0] - crop_size / 2)
    left = int(center[1] - crop_size / 2)

    return im[upper: upper + crop_size, left: left + crop_size, :]
