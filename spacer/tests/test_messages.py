import unittest
import json

from spacer.messages import ExtractFeaturesMsg


class TestExtractFeaturesMsg(unittest.TestCase):

    def test_serialize(self):

        msg = ExtractFeaturesMsg(
            pk=1,
            modelname='vgg16_coralnet_ver1',
            bucketname='spacer-test',
            storage_type='s3',
            imkey='edinburgh3.jpg',
            rowcols=[(100, 100)],
            outputkey='edinburgh3.jpg.feats'
        )

        print(ExtractFeaturesMsg.deserialize(msg.serialize()))
        print(msg)
        self.assertEqual(msg, ExtractFeaturesMsg.deserialize(msg.serialize()))

        self.assertEqual(msg, ExtractFeaturesMsg.deserialize(json.loads(json.dumps(msg.serialize()))))
