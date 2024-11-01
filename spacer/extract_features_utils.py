from __future__ import annotations

import numpy as np
from PIL import Image


def crop_patches(im: Image,
                 rowcols: list[tuple[int, int]],
                 crop_size: int) -> list[np.ndarray]:
    """
    Crop patches from an image
    :param im: image for cropping
    :param rowcols: [(row1, col1), (row2, col2), ...]
    :param crop_size: patch size
    :return: patch list
    """
    # Normalize to RGB mode (so for example, no transparency channel,
    # and not a monochrome format) to simplify later processing steps.
    im = im.convert('RGB')

    im = np.array(im)

    pad = crop_size
    im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    patches = [crop_simple(im, (row + pad, col + pad), crop_size)
               for row, col in rowcols]

    return patches


def crop_simple(im: np.ndarray,
                center: tuple[int, int],
                crop_size: int) -> np.ndarray:
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
