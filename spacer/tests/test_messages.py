import json
import unittest

from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    ClassifyImageMsg, \
    ClassifyReturnMsg, \
    TaskMsg, \
    TaskReturnMsg


class TestExtractFeaturesMsg(unittest.TestCase):

    def test_serialize(self):

        msg = ExtractFeaturesMsg.example()
        self.assertEqual(msg, ExtractFeaturesMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, ExtractFeaturesMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_missing_fields_in_serialized_message(self):

        data = ExtractFeaturesMsg.example().serialize()
        del data['job_token']  # Delete one of the keys.
        self.assertRaises(KeyError, ExtractFeaturesMsg.deserialize, data)

    def test_asserts(self):
        msg = ExtractFeaturesMsg.example()
        msg.feature_extractor_name = 'invalid_modelname'
        self.assertRaises(AssertionError,
                          ExtractFeaturesMsg.deserialize,
                          msg.serialize())

        msg = ExtractFeaturesMsg.example()
        msg.rowcols = []
        self.assertRaises(AssertionError,
                          ExtractFeaturesMsg.deserialize,
                          msg.serialize())

        msg = ExtractFeaturesMsg.example()
        msg.rowcols = [(120, 101, 121)]
        self.assertRaises(AssertionError,
                          ExtractFeaturesMsg.deserialize,
                          msg.serialize())

    def test_equal(self):
        msg1 = ExtractFeaturesMsg.example()
        msg2 = ExtractFeaturesMsg.example()
        msg3 = ExtractFeaturesMsg.example()
        msg3.imkey = 'different'
        self.assertEqual(msg1, msg2)
        self.assertNotEqual(msg1, msg3)


class TestExtractFeaturesReturnMsg(unittest.TestCase):

    def test_serialize(self):

        msg = ExtractFeaturesReturnMsg.example()
        self.assertEqual(msg, ExtractFeaturesReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, ExtractFeaturesReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))


class TestTrainClassifierMsg(unittest.TestCase):

    def test_serialize(self):

        msg = TrainClassifierMsg.example()
        self.assertEqual(msg, TrainClassifierMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, TrainClassifierMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))


class TestTrainClassifierReturnMsg(unittest.TestCase):

    def test_serialize(self):

        msg = TrainClassifierReturnMsg.example()
        self.assertEqual(msg, TrainClassifierReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, TrainClassifierReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))


class TestClassifyImageMsg(unittest.TestCase):

    def test_serialize(self):

        msg = ClassifyImageMsg.example()
        self.assertEqual(msg, ClassifyImageMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, ClassifyImageMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))


class TestDeployReturnMsg(unittest.TestCase):

    def test_serialize(self):

        msg = ClassifyReturnMsg.example()
        self.assertEqual(msg, ClassifyReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, ClassifyReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))


class TestTaskMsg(unittest.TestCase):

    def test_serialize_extract_features(self):

        task = ExtractFeaturesMsg.example()
        msg = TaskMsg(task='extract_features', payload=task)
        self.assertEqual(msg, TaskMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, TaskMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_train_classifier(self):

        task = TrainClassifierMsg.example()
        msg = TaskMsg(task='train_classifier', payload=task)
        self.assertEqual(msg, TaskMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, TaskMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_deploy(self):

        task = DeployMsg.example()
        msg = TaskMsg(task='deploy', payload=task)
        self.assertEqual(msg, TaskMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, TaskMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))


class TestTaskReturnMsg(unittest.TestCase):

    def test_serialize_extract_features(self):

        task = ExtractFeaturesMsg.example()
        org_msg = TaskMsg(task='extract_features', payload=task)

        return_task = ExtractFeaturesReturnMsg.example()
        msg = TaskReturnMsg(
            original_job=org_msg,
            ok=True,
            results=return_task,
            error_message=None
        )
        self.assertEqual(msg, TaskReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, TaskReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_train_classifier(self):

        task = TrainClassifierMsg.example()
        org_msg = TaskMsg(task='train_classifier', payload=task)

        return_task = TrainClassifierReturnMsg.example()
        msg = TaskReturnMsg(
            original_job=org_msg,
            ok=True,
            results=return_task,
            error_message=None
        )
        self.assertEqual(msg, TaskReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, TaskReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_deploy(self):
        msg = TaskReturnMsg.example()
        self.assertEqual(msg, TaskReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, TaskReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_error(self):
        task = DeployMsg.example()
        org_msg = TaskMsg(task='deploy', payload=task)

        msg = TaskReturnMsg(
            original_job=org_msg,
            ok=False,
            results=None,
            error_message='some error message'
        )
        self.assertEqual(msg, TaskReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, TaskReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))


