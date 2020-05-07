import unittest

import numpy as np
from PIL import Image

from spacer.extract_features_utils import gray2rgb, crop_patches


class TestGray2RGB(unittest.TestCase):

    def test_default(self):
        out_arr = gray2rgb(np.array(Image.new('L', (200, 200))))
        out_im = Image.fromarray(out_arr)
        self.assertEqual(out_im.mode, "RGB")


class TestCropPatch(unittest.TestCase):

    def test_rgb(self):
        rowcols = [(190, 226), (25, 359)]
        crop_size = 224
        patch_list = crop_patches(
            im=Image.new('RGB', (600, 600)),
            rowcols=rowcols,
            crop_size=crop_size
        )
        self.assertEqual(len(patch_list), len(rowcols))
        self.assertEqual(patch_list[0].shape[0], crop_size)
        self.assertEqual(patch_list[0].shape[1], crop_size)

    def test_gray(self):
        rowcols = [(190, 226), (25, 359)]
        crop_size = 224
        patch_list = crop_patches(
            im=Image.new('L', (600, 600)),
            rowcols=rowcols,
            crop_size=crop_size
        )
        self.assertEqual(Image.fromarray(patch_list[0]).mode, "RGB")
        self.assertEqual(len(patch_list), len(rowcols))
        self.assertEqual(patch_list[0].shape[0], crop_size)
        self.assertEqual(patch_list[0].shape[1], crop_size)


if __name__ == '__main__':
    unittest.main()
