from typing import List, Tuple
from PIL import Image


def check_rowcols(rowcols: List[Tuple[int, int]],
                  image: Image):

    im_width, im_height = image.size

    for row, col in rowcols:
        assert(isinstance(row, int)), \
            "Rows must be integers. Given was: [{}]".format(row)

        assert (isinstance(col, int)), \
            "Columns must be integers. Given was: [{}]".format(col)

        assert (row >= 0), \
            "Rows must be non-negative. Given was: [{}]".format(row)

        assert (col >= 0), \
            "Columns must be non-negative. Given was: [{}]".format(col)

        assert row < im_height, \
            "Roe {} outside image with nrows: {}".format(col, im_height)

        assert col < im_width, \
            "Column {} outside image with ncols: {}".format(col, im_width)



