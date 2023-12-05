from __future__ import annotations

from PIL import Image

from spacer import config
from spacer.exceptions import DataLimitError, RowColumnInvalidError


def check_extract_inputs(image: Image,
                         rowcols: list[tuple[int, int]],
                         im_key: str):

    im_width, im_height = image.size

    if im_width * im_height > config.MAX_IMAGE_PIXELS:
        raise DataLimitError(
            f"Image {im_key} has {im_width} x {im_height}"
            f" = {im_width * im_height} total pixels, which is larger"
            f" than the max allowed of {config.MAX_IMAGE_PIXELS}."
        )
    if len(rowcols) > config.MAX_POINTS_PER_IMAGE:
        raise DataLimitError(
            f"{len(rowcols)} point locations were specified for image"
            f" {im_key}, and that's larger than the max allowed of"
            f" {config.MAX_POINTS_PER_IMAGE}."
        )

    for row, col in rowcols:

        if not isinstance(row, int):
            raise RowColumnInvalidError(
                f"{im_key}: Row values must be integers."
                f" Given value was: {row}")
        if not isinstance(col, int):
            raise RowColumnInvalidError(
                f"{im_key}: Column values must be integers."
                f" Given value was: {col}")

        if row < 0:
            raise RowColumnInvalidError(
                f"{im_key}: Row values must be non-negative."
                f" Given value was: {row}")
        if col < 0:
            raise RowColumnInvalidError(
                f"{im_key}: Column values must be non-negative."
                f" Given value was: {col}")

        if row >= im_height:
            raise RowColumnInvalidError(
                f"{im_key}: Row value {row} falls outside this image's"
                f" valid range of 0-{im_height - 1}.")
        if col >= im_width:
            raise RowColumnInvalidError(
                f"{im_key}: Column value {col} falls outside this image's"
                f" valid range of 0-{im_width - 1}.")
