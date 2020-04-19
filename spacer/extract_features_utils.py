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


def crop_patch(im, rowcols, crop_size, scale, offsets=None):
    """
    Crop patches from an image
    :param im: image for cropping
    :param rowcols: [(row1, col1), (row2, col2), ...]
    :param crop_size: patch size
    :param scale: only supports scale = 1
    :param offsets: offset to the original (row, col)
    :return: patch list
    """

    assert scale == 1, "Only supports scale == 1"

    if offsets is None:
        offsets = np.zeros([len(rowcols), 2])

    # Ref: https://github.com/numpy/numpy/issues/11629
    # Looks like it's PIL issue
    _ = np.array(im)  # For some images np.array returns an empty array.
    im = np.array(im)  # Running it twice fixes this. Don't ask me why.

    if len(im.shape) == 2 or im.shape[2] == 1:
        im = gray2rgb(im)
    im = im[:, :, :3]  # only keep the first three color channels

    patchlist = []
    pad = crop_size

    im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    for ((row, col), offset) in zip(rowcols, offsets):
        center_org = np.asarray([row, col])
        center = np.round(pad + (center_org * scale) + offset).astype(np.int)

        patch = crop_simple(im, center, crop_size)

        patchlist.append(patch)

    return patchlist


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
