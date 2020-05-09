from typing import List, Tuple

import numpy as np
from PIL import Image


def gray2rgb(im: np.ndarray) -> np.ndarray:
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


def crop_patches(im: Image,
                 rowcols: List[Tuple[int, int]],
                 crop_size: int) -> List[np.ndarray]:
    """
    Crop patches from an image
    :param im: image for cropping
    :param rowcols: [(row1, col1), (row2, col2), ...]
    :param crop_size: patch size
    :return: patch list
    """

    # Ref: https://github.com/numpy/numpy/issues/11629
    # Looks like it's PIL issue
    _ = np.array(im)  # For some images np.array returns an empty array.
    im = np.array(im)  # Running it twice fixes this. Don't ask me why.

    if len(im.shape) == 2 or im.shape[2] == 1:
        im = gray2rgb(im)
    im = im[:, :, :3]  # only keep the first three color channels

    pad = crop_size
    im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    return [crop_simple(im, (row + pad, col + pad), crop_size)
            for row, col in rowcols]


def crop_simple(im: np.ndarray,
                center: Tuple[int, int],
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


def crop_patches_pil(im: Image,
                     rowcols: List[Tuple[int, int]],
                     crop_size: int):
    pad = crop_size
    image = Image.fromarray(
        np.pad(im.convert('RGB'),
               ((pad, pad), (pad, pad), (0, 0)), mode='reflect'),
        'RGB'
    )

    return [image.crop((pad + col - crop_size / 2,
                        pad + row - crop_size / 2,
                        pad + col + crop_size / 2,
                        pad + row + crop_size / 2))
            for row, col in rowcols]
