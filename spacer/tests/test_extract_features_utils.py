import unittest

import numpy as np
from PIL import Image

from io import BytesIO
from spacer import config
from spacer.storage import load_image
from spacer.messages import DataLocation
from spacer.extract_features_utils import gray2rgb, crop_patch


class TestGray2RGB(unittest.TestCase):

    def test_default(self):
        out_arr = gray2rgb(np.array(Image.new('L', (200, 200))))
        out_im = Image.fromarray(out_arr)
        self.assertEqual(out_im.mode, "RGB")


class TestCropPatch(unittest.TestCase):

    def test_rgb(self):
        rowcols = [(190, 226), (25, 359)]
        crop_size = 224
        patch_list = crop_patch(
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
        patch_list = crop_patch(
            im=Image.new('L', (600, 600)),
            rowcols=rowcols,
            crop_size=crop_size
        )
        self.assertEqual(Image.fromarray(patch_list[0]).mode, "RGB")
        self.assertEqual(len(patch_list), len(rowcols))
        self.assertEqual(patch_list[0].shape[0], crop_size)
        self.assertEqual(patch_list[0].shape[1], crop_size)

    def test_real_image(self):
        rowcols = [(20, 265),
                   (76, 295),
                   (59, 274),
                   (151, 62),
                   (265, 234)]
        crop_size = 224
        image_loc = DataLocation(storage_type='s3',
                                 key='08bfc10v7t.png',
                                 bucket_name='spacer-test')
        img = load_image(image_loc)
        patch_list = crop_patch(
            im=img,
            rowcols=rowcols,
            crop_size=crop_size
        )

        conn = config.get_s3_conn()
        bucket = conn.get_bucket('spacer-test', validate=True)
        npy_key = bucket.get_key('legacy_cropped.npy')
        legacy_npy = np.load(BytesIO(npy_key.get_contents_as_string()))

        self.assertTrue(np.allclose(legacy_npy, np.array(patch_list)))


class TestOpenImage(unittest.TestCase):

    def test_open(self):
        conn = config.get_s3_conn()
        bucket = conn.get_bucket('spacer-test', validate=True)
        npy_key = bucket.get_key('legacy_np.npy')
        legacy_npy = np.load(BytesIO(npy_key.get_contents_as_string()))

        image_loc = DataLocation(storage_type='s3',
                                 key='08bfc10v7t.png',
                                 bucket_name='spacer-test')
        img_npy = np.array(load_image(image_loc))[:, :, :3]

        self.assertTrue(np.allclose(legacy_npy, img_npy))


if __name__ == '__main__':
    unittest.main()
