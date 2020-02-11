import json
import unittest

from spacer import config
from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    ImageFeatures
from spacer.storage import storage_factory
from spacer.tasks import extract_features


class TestDummyExtractor(unittest.TestCase):

    def test_simple(self):
        msg = ExtractFeaturesMsg(
            pk=1,
            modelname='dummy',
            bucketname='spacer-test',
            storage_type='s3',
            imkey='edinburgh3.jpg',
            rowcols=[(100, 100)],
            outputkey='edinburgh3.jpg.feats'
        )

        return_msg = extract_features(msg)

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))

        # Now check that the features were stored as expected.
        storage = storage_factory(msg.storage_type, msg.bucketname)

        feats = ImageFeatures.deserialize(json.loads(
            storage.load_string(msg.outputkey)))

        self.assertEqual(feats.point_features[0].row, 100)
        self.assertEqual(feats.point_features[0].col, 100)


class TestCaffeExtractor(unittest.TestCase):

    @unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
    @unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
    def test_simple(self):

        msg = ExtractFeaturesMsg(
            pk=1,
            modelname='vgg16_coralnet_ver1',
            bucketname='spacer-test',
            storage_type='s3',
            imkey='edinburgh3.jpg',
            rowcols=[(100, 100)],
            outputkey='edinburgh3.jpg.feats'
        )

        return_msg = extract_features(msg)

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))

        # Now check that the features were stored as expected.
        storage = storage_factory(msg.storage_type, msg.bucketname)

        feats = ImageFeatures.deserialize(json.loads(
            storage.load_string(msg.outputkey)))

        self.assertEqual(feats.point_features[0].row, 100)
        self.assertEqual(feats.point_features[0].col, 100)

    @unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
    @unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
    def test_corner_case1(self):
        """
        This particular image caused trouble on the production server.
        The image file itself is lightly corrupted, and PIL doesn't quite like it.
        """

        msg = ExtractFeaturesMsg(
            pk=1,
            modelname='vgg16_coralnet_ver1',
            bucketname='spacer-test',
            storage_type='s3',
            imkey='kh6dydiix0.jpeg',
            rowcols=[(148, 50), (60, 425)],
            outputkey='kh6dydiix0.jpeg.feats'
        )
        storage = storage_factory('s3', 'spacer-test')
        storage.delete(msg.outputkey)
        return_msg = extract_features(msg)

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(storage.exists(msg.outputkey))

    @unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
    @unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
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
            rowcols=[(190, 226), (25, 359)],
            outputkey='sfq2mr5qbs.jpeg.feats'
        )
        storage = storage_factory('s3', 'spacer-test')
        storage.delete(msg.outputkey)
        return_msg = extract_features(msg)

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(storage.exists(msg.outputkey))


if __name__ == '__main__':
    unittest.main()