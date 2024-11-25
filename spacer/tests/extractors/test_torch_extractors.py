import unittest

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from spacer.extractors import FeatureExtractor
from spacer.extractors.torch_extractors import extract_feature, transformation
from ..common import TEST_EXTRACTORS
from ..decorators import require_test_extractors


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


@require_test_extractors
class TestExtractFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extractor = FeatureExtractor.deserialize(
            TEST_EXTRACTORS['efficientnet-b0'])

    def test_rgb(self):

        weights_datastream, _ = self.extractor.load_datastream('weights')
        torch_params = {'model_type': 'efficientnet',
                        'model_name': 'efficientnet-b0',
                        'weights_datastream': weights_datastream,
                        'num_class': 1275,
                        'crop_size': 224,
                        'batch_size': 10}
        patch_list = [np.array(Image.new('RGB', (224, 224))),
                      np.array(Image.new('RGB', (224, 224))),
                      np.array(Image.new('RGB', (224, 224)))]
        feats = extract_feature(patch_list=patch_list,
                                pyparams=torch_params)
        self.assertEqual(len(feats), len(patch_list))
        self.assertEqual(len(feats[0]), 1280)


if __name__ == '__main__':
    unittest.main()
