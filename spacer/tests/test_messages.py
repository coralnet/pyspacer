import json
import unittest

from spacer.messages import \
    DataLocation, \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    ClassifyFeaturesMsg, \
    ClassifyImageMsg, \
    ClassifyReturnMsg, \
    JobMsg, \
    JobReturnMsg


class TestDataLocation(unittest.TestCase):

    def test_serialize(self):

        msg = DataLocation.example()
        self.assertEqual(msg, DataLocation.deserialize(
            msg.serialize()))
        self.assertEqual(msg, DataLocation.deserialize(
            json.loads(json.dumps(msg.serialize()))))


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


class TestClassifyFeaturesMsg(unittest.TestCase):

    def test_serialize(self):

        msg = ClassifyFeaturesMsg.example()
        self.assertEqual(msg, ClassifyFeaturesMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, ClassifyFeaturesMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))


class TestClassifyReturnMsg(unittest.TestCase):

    def test_serialize(self):

        msg = ClassifyReturnMsg.example()
        self.assertEqual(msg, ClassifyReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, ClassifyReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))


class TestJobMsg(unittest.TestCase):

    def test_serialize_extract_features(self):

        task = ExtractFeaturesMsg.example()
        msg = JobMsg(task_name='extract_features', tasks=[task])
        self.assertEqual(msg, JobMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, JobMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_train_classifier(self):

        task = TrainClassifierMsg.example()
        msg = JobMsg(task_name='train_classifier', tasks=[task])
        self.assertEqual(msg, JobMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, JobMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_classify_features(self):

        task = ClassifyFeaturesMsg.example()
        msg = JobMsg(task_name='classify_features', tasks=[task])
        self.assertEqual(msg, JobMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, JobMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_classify_image(self):

        task = ClassifyImageMsg.example()
        msg = JobMsg(task_name='classify_image', tasks=[task])
        self.assertEqual(msg, JobMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, JobMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_many(self):

        task = ClassifyImageMsg.example()
        msg = JobMsg(task_name='classify_image',
                     tasks=[task, task, task])
        self.assertEqual(msg, JobMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, JobMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))


class TestJobReturnMsg(unittest.TestCase):

    def test_serialize_extract_features(self):

        task = ExtractFeaturesMsg.example()
        org_msg = JobMsg(task_name='extract_features', tasks=[task])

        return_task = ExtractFeaturesReturnMsg.example()
        msg = JobReturnMsg(
            original_job=org_msg,
            ok=True,
            results=[return_task],
            error_message=None
        )
        self.assertEqual(msg, JobReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, JobReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_train_classifier(self):

        task = TrainClassifierMsg.example()
        org_msg = JobMsg(task_name='train_classifier', tasks=[task])

        return_task = TrainClassifierReturnMsg.example()
        msg = JobReturnMsg(
            original_job=org_msg,
            ok=True,
            results=[return_task],
            error_message=None
        )
        self.assertEqual(msg, JobReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, JobReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_classify(self):
        msg = JobReturnMsg.example()
        self.assertEqual(msg, JobReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, JobReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_many_tasks(self):

        task = TrainClassifierMsg.example()
        org_msg = JobMsg(task_name='train_classifier',
                         tasks=[task, task, task, task])

        return_task = TrainClassifierReturnMsg.example()
        msg = JobReturnMsg(
            original_job=org_msg,
            ok=True,
            results=[return_task],
            error_message=None
        )
        self.assertEqual(msg, JobReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, JobReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_serialize_error(self):
        task = ClassifyImageMsg.example()
        org_msg = JobMsg(task_name='classify_image', tasks=[task])

        msg = JobReturnMsg(
            original_job=org_msg,
            ok=False,
            results=None,
            error_message='some error message'
        )
        self.assertEqual(msg, JobReturnMsg.deserialize(
            msg.serialize()))
        self.assertEqual(msg, JobReturnMsg.deserialize(
            json.loads(json.dumps(msg.serialize()))))
