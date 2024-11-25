import collections
import unittest

import torch

from spacer import extractors
from spacer.extractors.efficientnet_utils import \
    round_filters, \
    round_repeats, \
    drop_connect, \
    get_same_padding_conv2d, \
    Conv2dDynamicSamePadding, \
    BlockDecoder, \
    get_model_params


class TestGetModels(unittest.TestCase):

    def test_invalid_model(self):
        with self.assertRaises(NotImplementedError):
            _ = extractors.get_model(model_type='dummy',
                                     model_name='dummy',
                                     num_classes=1000)


class TestEfficientNet(unittest.TestCase):

    def test_efficientnet(self):
        model_param = {'model_type': 'efficientnet',
                       'model_name': 'efficientnet-b0',
                       'num_classes': 1000}
        net = extractors.get_model(model_type=model_param['model_type'],
                                   model_name=model_param['model_name'],
                                   num_classes=model_param['num_classes'])
        with torch.no_grad():
            output = net(torch.rand(1, 3, 224, 224))

        self.assertEqual(net.get_image_size(model_param['model_name']), 224)
        self.assertEqual(output.shape[1], model_param['num_classes'])
        with self.assertRaises(ValueError):
            net._check_model_name_is_valid(model_name='dummy')

    def test_get_model_params(self):
        with self.assertRaises(NotImplementedError):
            get_model_params(model_name='dummy',
                             override_params=None)


class TestEfficientNetUtils(unittest.TestCase):

    GlobalParams = collections.namedtuple(
        'GlobalParams', ['width_coefficient', 'depth_coefficient',
                         'depth_divisor', 'min_depth']
    )

    def test_round_filter(self):
        global_params = self.GlobalParams(
            width_coefficient=None,
            depth_coefficient=0.1,
            depth_divisor=8,
            min_depth=None
        )
        num_filter = round_filters(32, global_params)
        self.assertEqual(num_filter, 32)

        global_params = self.GlobalParams(
            width_coefficient=1,
            depth_coefficient=0.1,
            depth_divisor=32,
            min_depth=None
        )
        num_filter = round_filters(40, global_params)
        self.assertEqual(num_filter, 64)

    def test_round_repeat(self):
        global_params = self.GlobalParams(
            width_coefficient=1,
            depth_coefficient=None,
            depth_divisor=8,
            min_depth=None
        )
        num_repeat = round_repeats(8, global_params)
        self.assertEqual(num_repeat, 8)

    def test_drop_connect(self):
        input_ = torch.rand(1, 10, 5, 5)
        output1 = drop_connect(input_, p=0.5, training=False)
        output2 = drop_connect(input_, p=0.999, training=True)

        self.assertListEqual(input_.tolist(), output1.tolist())
        self.assertAlmostEqual(torch.sum(output2).item(), 0)

    def test_get_same_padding_conv2d(self):
        cls = get_same_padding_conv2d(image_size=None)

        self.assertEqual(cls.__name__, 'Conv2dDynamicSamePadding')


class TestConv2dDynamicSamePadding(unittest.TestCase):

    def test_dynamic_padding(self):
        conv = Conv2dDynamicSamePadding(in_channels=2,
                                        out_channels=4,
                                        kernel_size=3)
        _ = conv(torch.rand(1, 2, 10, 10))

        self.assertEqual(conv.stride, (1, 1))


class TestBlockDecoder(unittest.TestCase):

    def test_encode(self):
        BlockArgs = collections.namedtuple('BlockArgs', [
            'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
            'expand_ratio', 'id_skip', 'strides', 'se_ratio'])
        block_params = [BlockArgs(
            num_repeat=1,
            kernel_size=3,
            input_filters=32,
            output_filters=16,
            expand_ratio=1,
            id_skip=False,
            strides=(1, 1),
            se_ratio=0.25
        )]
        block_strings = BlockDecoder.encode(block_params)
        self.assertListEqual(block_strings,
                             ['r1_k3_s11_e1_i32_o16_se0.25_noskip'])


if __name__ == '__main__':
    unittest.main()
