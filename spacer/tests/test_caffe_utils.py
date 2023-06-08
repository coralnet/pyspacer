import time
import unittest

import numpy as np
from PIL import Image

from spacer import config
from spacer.storage import download_model


@unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
class TestTransformer(unittest.TestCase):

    def test_process(self):
        from spacer.caffe_utils import Transformer
        trans = Transformer()
        im_pil = Image.new('RGB', (50, 50))
        im_arr = np.asarray(im_pil)
        im_arr2 = trans.deprocess(trans.preprocess(im_arr))
        self.assertTrue(np.array_equal(im_arr, im_arr2))


@unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
@unittest.skipUnless(config.HAS_S3_MODEL_ACCESS, 'Need model access')
class TestClassifyFromPatchList(unittest.TestCase):

    def setUp(self):
        config.filter_warnings()
        self.modeldef_path, _ = download_model(
            'vgg16_coralnet_ver1.deploy.prototxt')
        self.modelweighs_path, self.model_was_cached = download_model(
            'vgg16_coralnet_ver1.caffemodel')

    def test_rgb(self):
        from spacer.caffe_utils import classify_from_patchlist

        caffe_params = {'im_mean': [128, 128, 128],
                        'scaling_method': 'scale',
                        'crop_size': 224,
                        'batch_size': 10}

        feats = classify_from_patchlist(
            [np.array(Image.new('RGB', (224, 224)))],
            caffe_params,
            self.modeldef_path,
            self.modelweighs_path,
            scorelayer='fc7')
        self.assertEqual(len(feats), 1)
        self.assertEqual(len(feats[0]), 4096)

    def test_net_caching(self):
        """ Call classify_from_patchlist twice to check if the LRU caching on
        load_net method works
        """
        from spacer.caffe_utils import load_net

        # Clear cache to make sure it's not set from previous test.
        load_net.cache_clear()
        t0 = time.time()
        _ = load_net(self.modeldef_path, self.modelweighs_path)
        t1 = time.time() - t0

        t0 = time.time()
        _ = load_net(self.modeldef_path, self.modelweighs_path)
        t2 = time.time() - t0
        self.assertLess(t2, t1)


if __name__ == '__main__':
    unittest.main()
