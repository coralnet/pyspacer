import unittest

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from spacer import config
from spacer.storage import download_model
from spacer.torch_utils import extract_feature
from spacer.torch_utils import transformation


@unittest.skipUnless(config.HAS_S3_MODEL_ACCESS, 'No access to models')
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


@unittest.skipUnless(config.HAS_S3_MODEL_ACCESS, 'No access to models')
class TestExtractFeatures(unittest.TestCase):

    def setUp(self):
        self.modelweighs_path, self.model_was_cashed = download_model(
            'efficientnetb0_5eps_best.pt')

    def test_rgb(self):

        torch_params = {'model_type': 'efficientnet',
                        'model_name': 'efficientnet-b0',
                        'weights_path': self.modelweighs_path,
                        'num_class': 1279,
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
