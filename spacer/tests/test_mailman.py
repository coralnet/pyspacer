import os
import unittest

from spacer import config
from spacer.tasks import process_job
from spacer.messages import \
    JobMsg, \
    JobReturnMsg, \
    ExtractFeaturesMsg, \
    TrainClassifierMsg, \
    ClassifyImageMsg, \
    DataLocation, \
    ImageLabels

TEST_URL = \
    'https://upload.wikimedia.org/wikipedia/commons/7/7b/Red_sea_coral_reef.jpg'
TEST_URL_FILENAME = 'Red_sea_coral_reef.jpg'


class TestProcessJobErrorHandling(unittest.TestCase):

    def setUp(self):
        config.filter_warnings()

    def tearDown(self):
        if os.path.exists(TEST_URL_FILENAME):
            os.remove(TEST_URL_FILENAME)

    def test_input_type(self):
        self.assertRaises(AssertionError, process_job, 'sdf')

    def test_task_name(self):
        msg = JobMsg.example()
        msg.task_name = 'invalid'
        self.assertRaises(AssertionError, process_job, msg)

    def test_feature_extract(self):
        msg = JobMsg(task_name='extract_features',
                     tasks=[ExtractFeaturesMsg(
                         job_token='test_job',
                         feature_extractor_name='dummy',
                         rowcols=[(1, 1)],
                         image_loc=DataLocation(storage_type='memory',
                                                key='missing_image'),
                         feature_loc=DataLocation(storage_type='memory',
                                                  key='doesnt_matter'))])
        return_msg = process_job(msg)
        self.assertFalse(return_msg.ok)
        self.assertIn("KeyError", return_msg.error_message)
        self.assertIn("missing_image", return_msg.error_message)
        self.assertEqual(type(return_msg), JobReturnMsg)

    def test_img_classify_bad_url(self):

        msg = JobMsg(task_name='classify_image',
                     tasks=[ClassifyImageMsg(
                         job_token='my_job',
                         image_loc=DataLocation(storage_type='url',
                                                key='http://invalid_url.com'),
                         feature_extractor_name='invalid_name',
                         rowcols=[(1, 1)],
                         classifier_loc=DataLocation(storage_type='memory',
                                                     key='doesnt_matter'))])
        return_msg = process_job(msg)
        self.assertFalse(return_msg.ok)
        self.assertIn("URLError", return_msg.error_message)
        self.assertIn("SpacerInputError", return_msg.error_message)
        self.assertEqual(type(return_msg), JobReturnMsg)

    def test_img_classify_feature_extractor_name(self):

        msg = JobMsg(task_name='classify_image',
                     tasks=[ClassifyImageMsg(
                         job_token='my_job',
                         image_loc=DataLocation(storage_type='url',
                                                key=TEST_URL),
                         feature_extractor_name='invalid_name',
                         rowcols=[(1, 1)],
                         classifier_loc=DataLocation(storage_type='memory',
                                                     key='doesnt_matter'))])
        return_msg = process_job(msg)
        self.assertFalse(return_msg.ok)
        self.assertIn("AssertionError", return_msg.error_message)
        self.assertIn("Model name invalid_name not registered",
                      return_msg.error_message)
        self.assertEqual(type(return_msg), JobReturnMsg)

    def test_img_classify_classifier_key(self):

        msg = JobMsg(task_name='classify_image',
                     tasks=[ClassifyImageMsg(
                         job_token='my_job',
                         image_loc=DataLocation(storage_type='url',
                                                key=TEST_URL),
                         feature_extractor_name='dummy',
                         rowcols=[(1, 1)],
                         classifier_loc=DataLocation(storage_type='memory',
                                                     key='no_classifier_here')
                     )])
        return_msg = process_job(msg)
        self.assertFalse(return_msg.ok)
        self.assertIn("KeyError", return_msg.error_message)
        self.assertIn("no_classifier_here", return_msg.error_message)
        self.assertEqual(type(return_msg), JobReturnMsg)

    def test_train_classifier(self):

        for clf_type in config.CLASSIFIER_TYPES:
            msg = JobMsg(task_name='train_classifier',
                         tasks=[TrainClassifierMsg(
                             job_token='my_job',
                             trainer_name='minibatch',
                             nbr_epochs=1,
                             clf_type=clf_type,
                             train_labels=ImageLabels(data={
                                 'my_feats': [(1, 1, 1), (2, 2, 2)]
                             }),
                             val_labels=ImageLabels(data={
                                 'my_feats': [(1, 1, 1), (2, 2, 2)]
                             }),
                             features_loc=DataLocation(storage_type='memory',
                                                       key='my_feats'),
                             previous_model_locs=[
                                 DataLocation(storage_type='memory',
                                              key='my_previous_model')
                             ],
                             model_loc=DataLocation(storage_type='memory',
                                                    key='model'),
                             valresult_loc=DataLocation(storage_type='memory',
                                                        key='val_res'))])

            return_msg = process_job(msg)
            self.assertFalse(return_msg.ok)
            self.assertIn("KeyError", return_msg.error_message)
            self.assertIn("my_previous_model", return_msg.error_message)
            self.assertEqual(type(return_msg), JobReturnMsg)


class TestProcessJobMultiple(unittest.TestCase):

    def setUp(self):
        config.filter_warnings()

    def tearDown(self):
        if os.path.exists(TEST_URL_FILENAME):
            os.remove(TEST_URL_FILENAME)

    def test_multiple_feature_extract(self):
        extract_msg = ExtractFeaturesMsg(
            job_token='test_job',
            feature_extractor_name='dummy',
            rowcols=[(1, 1)],
            image_loc=DataLocation(storage_type='url',
                                   key=TEST_URL),
            feature_loc=DataLocation(storage_type='memory',
                                     key='my_feats'))

        job_msg = JobMsg(task_name='extract_features',
                         tasks=[extract_msg, extract_msg])

        return_msg = process_job(job_msg)
        self.assertTrue(return_msg.ok)
        self.assertEqual(len(return_msg.results), 2)
        self.assertEqual(type(return_msg), JobReturnMsg)

    def test_multiple_feature_extract_one_fail(self):
        good_extract_msg = ExtractFeaturesMsg(
            job_token='test_job',
            feature_extractor_name='dummy',
            rowcols=[(1, 1)],
            image_loc=DataLocation(storage_type='url',
                                   key=TEST_URL),
            feature_loc=DataLocation(storage_type='memory',
                                     key='my_feats'))

        fail_extract_msg = ExtractFeaturesMsg(
            job_token='test_job',
            feature_extractor_name='dummy',
            rowcols=[(1, 1)],
            image_loc=DataLocation(storage_type='url',
                                   key='bad_url'),
            feature_loc=DataLocation(storage_type='memory',
                                     key='my_feats'))

        job_msg = JobMsg(task_name='extract_features',
                         tasks=[good_extract_msg,
                                fail_extract_msg])

        return_msg = process_job(job_msg)
        self.assertFalse(return_msg.ok)
        self.assertIn('bad_url', return_msg.error_message)
        self.assertEqual(type(return_msg), JobReturnMsg)


if __name__ == '__main__':
    unittest.main()
