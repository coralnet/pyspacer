import torch
import unittest
from PIL import Image
from torchvision import transforms

import numpy as np

from spacer import config
from spacer.storage import download_model


@unittest.skipUnless(config.HAS_TORCH, 'Pytorch not installed')
class TestTransformation(unittest.TestCase):

    def test_transformer(self):
        from spacer.torch_utils import transformation
        test_channels = 3
        height, width = 4, 4
        transformer = transformation()

        input_data = torch.ByteTensor(test_channels, height, width).random_(0, 255).float().div_(255)
        img = transforms.ToPILImage()(input_data)
        output = transformer(img)
        self.assertTrue(np.allclose(input_data.numpy(), output.numpy()))

        ndarray = np.random.randint(low=0, high=255, size=(height, width, test_channels)).astype(np.uint8)
        output = transformer(ndarray)
        expected_output = ndarray.transpose((2, 0, 1)) / 255.0
        self.assertTrue(np.allclose(output.numpy(), expected_output))

        ndarray = np.random.rand(height, width, test_channels).astype(np.float32)
        output = transformer(ndarray)
        expected_output = ndarray.transpose((2, 0, 1))
        self.assertTrue(np.allclose(output.numpy(), expected_output))


@unittest.skipUnless(config.HAS_TORCH, 'Pytorch not installed')
class TestClassifyFromPatchList(unittest.TestCase):

    def setUp(self):
        self.modelweighs_path, self.model_was_cashed = download_model(
            'efficientnetb0_5eps_best.pt')

    def test_rgb(self):
        from spacer.torch_utils import classify_from_patchlist

        torch_params = {'model_type': 'efficientnet',
                        'model_name': 'efficientnet-b0',
                        'weights_path': self.modelweighs_path,
                        'num_class': 1279,
                        'crop_size': 224,
                        'batch_size': 10}
        _, feats = classify_from_patchlist(Image.new('RGB', (600, 600)),
                                           [(300, 300, 1)], torch_params)
        self.assertEqual(len(feats), 1)
        self.assertEqual(len(feats[0]), 1280)

    def test_gray(self):
        from spacer.torch_utils import classify_from_patchlist

        torch_params = {'model_type': 'efficientnet',
                        'model_name': 'efficientnet-b0',
                        'weights_path': self.modelweighs_path,
                        'num_class': 1279,
                        'crop_size': 224,
                        'batch_size': 10}
        _, feats = classify_from_patchlist(Image.new('L', (600, 600)),
                                           [(300, 300, 1)], torch_params)
        self.assertEqual(len(feats), 1)
        self.assertEqual(len(feats[0]), 1280)


@unittest.skipUnless(config.HAS_TORCH, 'Pytorch not installed')
class TestGray2RGB(unittest.TestCase):

    def test_nominal(self):
        from spacer.torch_utils import gray2rgb
        out_arr = gray2rgb(np.array(Image.new('L', (200, 200))))
        out_im = Image.fromarray(out_arr)
        self.assertEqual(out_im.mode, "RGB")


if __name__ == '__main__':
    unittest.main()
