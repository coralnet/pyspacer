from io import BytesIO
import unittest

import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import transforms

from spacer.extractors import EfficientNetExtractor
from spacer.extractors.torch_extractors import transformation
from spacer.tests.utils import random_image


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


def random_patches_for_torch(num_patches: int, patch_size: int):
    transformer = transformation()
    return torch.stack([
        transformer(random_image(patch_size, patch_size))
        for _ in range(num_patches)
    ])


def run_epoch(model, optimizer, patches):
    _ = model.train()

    # This is all one batch. Could potentially change this to a loop for
    # multiple batches of patches.
    optimizer.zero_grad()
    model(patches)
    optimizer.step()

    _ = model.eval()


def sample_weights_trained(num_epochs, patches):
    """
    Generate weights by actually running network training.
    """
    model = EfficientNetExtractor.untrained_model()

    # Based on CoralNet parameters.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=3e-06,
    )

    # In practice there should be loss tracking, and deciding when to stop
    # running epochs should be based on that.
    epoch = 1
    while epoch <= num_epochs:
        run_epoch(model, optimizer, patches)
        epoch += 1

    return dict(
        net=model.state_dict(),
        # This value alone allows us to test everything that's needed from
        # add_safe_globals().
        epoch=np.int64(num_epochs),
        # CoralNet's weights also include an optimizer and scheduler,
        # but those aren't needed to test load_weights(), at least.
    )


class TestLoadWeights(unittest.TestCase):

    def test(self):
        patches = random_patches_for_torch(5, EfficientNetExtractor.CROP_SIZE)
        with BytesIO() as stream:
            torch.save(
                sample_weights_trained(num_epochs=3, patches=patches),
                stream,
            )
            stream.seek(0)
            # Just seeing whether this crashes or not is a good test.
            # Testing with different combos of package versions is important.
            net = EfficientNetExtractor.load_weights(stream)
        self.assertTrue(isinstance(net, nn.Module))


if __name__ == '__main__':
    unittest.main()
