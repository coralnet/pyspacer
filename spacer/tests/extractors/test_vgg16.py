import time
import unittest

import numpy as np
from PIL import Image

from spacer import config
from spacer.extractors import FeatureExtractor
from spacer.extractors.vgg16 import load_net, Transformer
from ..common import TEST_EXTRACTORS
from ..decorators import require_caffe, require_cn_test_extractors


class TestTransformer(unittest.TestCase):

    def test_process(self):
        trans = Transformer()
        im_pil = Image.new('RGB', (50, 50))
        im_arr = np.asarray(im_pil)
        im_arr2 = trans.deprocess(trans.preprocess(im_arr))
        self.assertTrue(np.array_equal(im_arr, im_arr2))


@require_caffe
@require_cn_test_extractors
class TestVGG16CaffeExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config.filter_warnings()

        cls.extractor = FeatureExtractor.deserialize(TEST_EXTRACTORS['vgg16'])
        cls.definition_filepath, _ = \
            cls.extractor.load_data_into_filesystem('definition')
        cls.weights_filepath, _ = \
            cls.extractor.load_data_into_filesystem('weights')

    def test_patches_to_features(self):
        crop_size = self.extractor.CROP_SIZE
        patch_list = [np.array(Image.new('RGB', (crop_size, crop_size))),
                      np.array(Image.new('RGB', (crop_size, crop_size))),
                      np.array(Image.new('RGB', (crop_size, crop_size)))]
        feats, remote_loaded = self.extractor.patches_to_features(
            patch_list=patch_list)
        self.assertEqual(len(feats), len(patch_list))
        self.assertEqual(len(feats[0]), 4096)

    def test_load_net_caching(self):
        """
        Call load_net() twice to check if the LRU caching
        on that method works.
        """
        # Clear cache to make sure it's not set from previous test.
        load_net.cache_clear()
        t0 = time.time()
        _ = load_net(self.definition_filepath, self.weights_filepath)
        t1 = time.time() - t0

        t0 = time.time()
        _ = load_net(self.definition_filepath, self.weights_filepath)
        t2 = time.time() - t0
        self.assertLess(t2, t1)


if __name__ == '__main__':
    unittest.main()
