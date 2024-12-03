import time
import unittest

import numpy as np
from PIL import Image

from spacer import config
from spacer.extractors import FeatureExtractor
from ..common import TEST_EXTRACTORS
from ..decorators import require_caffe, require_cn_test_extractors


@require_caffe
class TestTransformer(unittest.TestCase):

    def test_process(self):
        from spacer.extractors.vgg16 import Transformer
        trans = Transformer()
        im_pil = Image.new('RGB', (50, 50))
        im_arr = np.asarray(im_pil)
        im_arr2 = trans.deprocess(trans.preprocess(im_arr))
        self.assertTrue(np.array_equal(im_arr, im_arr2))


@require_caffe
@require_cn_test_extractors
class TestClassifyFromPatchList(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config.filter_warnings()

        extractor = FeatureExtractor.deserialize(TEST_EXTRACTORS['vgg16'])
        cls.definition_filepath, _ = \
            extractor.load_data_into_filesystem('definition')
        cls.weights_filepath, _ = \
            extractor.load_data_into_filesystem('weights')

    def test_rgb(self):
        from spacer.caffe_utils import classify_from_patchlist

        caffe_params = {'im_mean': [128, 128, 128],
                        'scaling_method': 'scale',
                        'crop_size': 224,
                        'batch_size': 10}

        feats = classify_from_patchlist(
            [np.array(Image.new('RGB', (224, 224)))],
            caffe_params,
            self.definition_filepath,
            self.weights_filepath,
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
        _ = load_net(self.definition_filepath, self.weights_filepath)
        t1 = time.time() - t0

        t0 = time.time()
        _ = load_net(self.definition_filepath, self.weights_filepath)
        t2 = time.time() - t0
        self.assertLess(t2, t1)


if __name__ == '__main__':
    unittest.main()
