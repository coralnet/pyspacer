import unittest

import numpy as np
from PIL import Image

from spacer import config
from spacer.caffe_utils import classify_from_patchlist, gray2rgb
from spacer.storage import download_model


@unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
class TestClassifyFromPatchList(unittest.TestCase):

    def setUp(self):
        self.modeldef_path, _ = download_model(
            'vgg16_coralnet_ver1.deploy.prototxt')
        self.modelweighs_path, self.model_was_cashed = download_model(
            'vgg16_coralnet_ver1.caffemodel')

    def test_rgb(self):

        caffe_params = {'im_mean': [128, 128, 128],
                        'scaling_method': 'scale',
                        'crop_size': 224,
                        'batch_size': 10}

        gt, est, feats = classify_from_patchlist(
            Image.new('RGB', (600, 600)),
            [(300, 300, 1)],
            caffe_params,
            self.modeldef_path,
            self.modelweighs_path,
            scorelayer='fc7')
        self.assertEqual(len(feats), 1)
        self.assertEqual(len(feats[0]), 4096)

    def test_gray(self):
        caffe_params = {'im_mean': [128, 128, 128],
                        'scaling_method': 'scale',
                        'crop_size': 224,
                        'batch_size': 10}

        gt, est, feats = classify_from_patchlist(
            Image.new('L', (600, 600)),
            [(300, 300, 1)],
            caffe_params,
            self.modeldef_path,
            self.modelweighs_path,
            scorelayer='fc7')
        self.assertEqual(len(feats), 1)
        self.assertEqual(len(feats[0]), 4096)


@unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
class TestGray2RGB(unittest.TestCase):

    def test_nominal(self):

        out_arr = gray2rgb(np.array(Image.new('L', (200, 200))))
        out_im = Image.fromarray(out_arr)
        self.assertEqual(out_im.mode, "RGB")


if __name__ == '__main__':
    unittest.main()


