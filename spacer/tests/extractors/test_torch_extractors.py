import unittest

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from spacer.extractors import EfficientNetExtractor
from spacer.extractors.torch_extractors import transformation


class TestTransformation(unittest.TestCase):

    def test_transformer(self):
        test_channels = 3
        height, width = 4, 4
        transformer = transformation()

        input_data = torch.ByteTensor(test_channels, height, width).\
            random_(0, 255).float().div_(255)
        img = transforms.ToPILImage()(input_data)
        output = transformer(img)
        self.assertTrue(np.allclose(input_data.numpy(), output.numpy()))

        ndarray = np.random.randint(
            low=0, high=255, size=(height, width, test_channels)).\
            astype(np.uint8)
        output = transformer(ndarray)
        expected_output = ndarray.transpose((2, 0, 1)) / 255.0
        self.assertTrue(np.allclose(output.numpy(), expected_output))

        ndarray = np.random.rand(height, width, test_channels).\
            astype(np.float32)
        output = transformer(ndarray)
        expected_output = ndarray.transpose((2, 0, 1))
        self.assertTrue(np.allclose(output.numpy(), expected_output))


class TestPatchesToFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extractor = EfficientNetExtractor.untrained_instance()

    def test(self):
        crop_size = self.extractor.CROP_SIZE
        patch_list = [np.array(Image.new('RGB', (crop_size, crop_size))),
                      np.array(Image.new('RGB', (crop_size, crop_size))),
                      np.array(Image.new('RGB', (crop_size, crop_size)))]
        feats, remote_loaded = self.extractor.patches_to_features(
            patch_list=patch_list)
        self.assertEqual(len(feats), len(patch_list))
        self.assertEqual(len(feats[0]), 1280)


if __name__ == '__main__':
    unittest.main()
