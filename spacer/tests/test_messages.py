import unittest
import json

from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    PointFeatures, \
    ImageFeatures


class TestExtractFeaturesMsg(unittest.TestCase):

    def test_serialize(self):

        msg = ExtractFeaturesMsg.example()
        self.assertEqual(msg, ExtractFeaturesMsg.deserialize(msg.serialize()))
        self.assertEqual(msg, ExtractFeaturesMsg.deserialize(json.loads(json.dumps(msg.serialize()))))

    def test_missing_fields_in_serialized_message(self):

        data = ExtractFeaturesMsg.example().serialize()
        del data['pk']  # Delete one of the keys.
        self.assertRaises(TypeError, ExtractFeaturesMsg.deserialize, data)

    def test_missing_storage_type_in_serialized_message(self):

        data = ExtractFeaturesMsg.example().serialize()
        del data['storage_type']  # Delete storage_key. Should still work since it is optional.
        msg = ExtractFeaturesMsg.deserialize(data)
        self.assertEqual(msg.storage_type, 's3')

    def test_asserts(self):
        msg = ExtractFeaturesMsg.example()
        msg.storage_type = 'invalid_storage'
        self.assertRaises(AssertionError, ExtractFeaturesMsg.deserialize, msg.serialize())

        msg = ExtractFeaturesMsg.example()
        msg.modelname = 'invalid_modelname'
        self.assertRaises(AssertionError, ExtractFeaturesMsg.deserialize, msg.serialize())

        msg = ExtractFeaturesMsg.example()
        msg.rowcols = []
        self.assertRaises(AssertionError, ExtractFeaturesMsg.deserialize, msg.serialize())

        msg = ExtractFeaturesMsg.example()
        msg.rowcols = [(120, 101, 121)]
        self.assertRaises(AssertionError, ExtractFeaturesMsg.deserialize, msg.serialize())


class TestExtractFeaturesReturnMsg(unittest.TestCase):

    def test_serialize(self):

        msg = ExtractFeaturesReturnMsg.example()
        self.assertEqual(msg, ExtractFeaturesReturnMsg.deserialize(msg.serialize()))
        self.assertEqual(msg, ExtractFeaturesReturnMsg.deserialize(json.loads(json.dumps(msg.serialize()))))


class TestTrainClassifierMsg(unittest.TestCase):

    def test_serialize(self):

        msg = TrainClassifierMsg.example()
        self.assertEqual(msg, TrainClassifierMsg.deserialize(msg.serialize()))
        self.assertEqual(msg, TrainClassifierMsg.deserialize(json.loads(json.dumps(msg.serialize()))))


class TestPointFeatures(unittest.TestCase):

    def test_serialize(self):

        msg = PointFeatures.example()
        self.assertEqual(msg, PointFeatures.deserialize(msg.serialize()))
        self.assertEqual(msg, PointFeatures.deserialize(json.loads(json.dumps(msg.serialize()))))


class TestImageFeatures(unittest.TestCase):

    def test_serialize(self):

        msg = ImageFeatures.example()
        msg2 = ImageFeatures.deserialize(msg.serialize())
        print(msg)
        print(msg2)
        print(msg2.point_features[0].__dict__)
        self.assertEqual(msg, msg2)
        #self.assertEqual(msg, ImageFeatures.deserialize(json.loads(json.dumps(msg.serialize()))))