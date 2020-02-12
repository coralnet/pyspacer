import json
import unittest
import warnings

from spacer import config
from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    ImageFeatures
from spacer.storage import storage_factory
from spacer.extract_features import DummyExtractor, VGG16CaffeExtractor


class TestDummyExtractor(unittest.TestCase):

    def test_simple(self):
        msg = ExtractFeaturesMsg(
            pk=1,
            modelname='dummy',
            bucketname='spacer-test',
            storage_type='local',
            imkey='not_used',
            rowcols=[[100, 100]],
            outputkey='not_used'
        )

        storage = storage_factory(msg.storage_type, msg.bucketname)
        ext = DummyExtractor(msg, storage)

        features, return_msg = ext(msg)

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(isinstance(features, ImageFeatures))

        # Check some feature metadata
        self.assertEqual(features.point_features[0].row, 100)
        self.assertEqual(features.point_features[0].col, 100)


class TestCaffeExtractor(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)

    @unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
    @unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to tests')
    @unittest.skipUnless(config.HAS_S3_MODEL_ACCESS, 'No access to models')
    def test_simple(self):

        msg = ExtractFeaturesMsg(
            pk=1,
            modelname='vgg16_coralnet_ver1',
            bucketname='spacer-test',
            storage_type='s3',
            imkey='edinburgh3.jpg',
            rowcols=[[100, 100]],
            outputkey='edinburgh3.jpg.feats'
        )

        storage = storage_factory(msg.storage_type, msg.bucketname)
        ext = VGG16CaffeExtractor(msg, storage)

        features, return_msg = ext(msg)

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(isinstance(features, ImageFeatures))

        # Check some feature metadata
        self.assertEqual(features.point_features[0].row, 100)
        self.assertEqual(features.point_features[0].col, 100)

    @unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
    @unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
    @unittest.skipUnless(config.HAS_S3_MODEL_ACCESS, 'No access to models')
    def test_corner_case1(self):
        """
        This particular image caused trouble on the production server.
        The image file itself is lightly corrupted, and PIL doesn't like it.
        """

        msg = ExtractFeaturesMsg(
            pk=1,
            modelname='vgg16_coralnet_ver1',
            bucketname='spacer-test',
            storage_type='s3',
            imkey='kh6dydiix0.jpeg',
            rowcols=[[148, 50], [60, 425]],
            outputkey='kh6dydiix0.jpeg.feats'
        )
        storage = storage_factory(msg.storage_type, msg.bucketname)
        ext = VGG16CaffeExtractor(msg, storage)

        features, return_msg = ext(msg)

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(isinstance(features, ImageFeatures))

        # Check some feature metadata
        self.assertEqual(features.point_features[0].row, 148)
        self.assertEqual(features.point_features[0].col, 50)

    @unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
    @unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
    @unittest.skipUnless(config.HAS_S3_MODEL_ACCESS, 'No access to models')
    def test_cornercase2(self):
        """
        This particular image caused trouble on the production server.
        The image file itself is lightly corrupted, and PIL doesn't
        quite like it.
        """
        msg = ExtractFeaturesMsg(
            pk=1,
            modelname='vgg16_coralnet_ver1',
            bucketname='spacer-test',
            storage_type='s3',
            imkey='sfq2mr5qbs.jpeg',
            rowcols=[[190, 226], [25, 359]],
            outputkey='sfq2mr5qbs.jpeg.feats'
        )
        storage = storage_factory(msg.storage_type, msg.bucketname)
        ext = VGG16CaffeExtractor(msg, storage)

        features, return_msg = ext(msg)

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(isinstance(features, ImageFeatures))

        # Check some feature metadata
        self.assertEqual(features.point_features[0].row, 190)
        self.assertEqual(features.point_features[0].col, 226)


if __name__ == '__main__':
    unittest.main()