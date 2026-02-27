import unittest

from spacer import config
from spacer.data_classes import DataLocation
from spacer.extractors import DummyExtractor
from spacer.messages import (
    ClassifyImageMsg,
    ExtractFeaturesMsg,
    JobMsg,
    JobReturnMsg,
    TrainClassifierMsg,
    TrainingTaskLabels,
)
from spacer.tasks import process_job

TEST_URL = \
    'https://www.gstatic.com/images/branding/searchlogo/ico/favicon.ico'


class TestProcessJobErrorHandling(unittest.TestCase):

    def setUp(self):
        config.filter_warnings()

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
                         extractor=DummyExtractor(),
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

        bad_url = 'http::/invalid_url.com'
        msg = JobMsg(task_name='classify_image',
                     tasks=[ClassifyImageMsg(
                         job_token='my_job',
                         image_loc=DataLocation(storage_type='url',
                                                key=bad_url),
                         extractor=DummyExtractor(),
                         rowcols=[(1, 1)],
                         classifier_loc=DataLocation(storage_type='memory',
                                                     key='doesnt_matter'))])
        return_msg = process_job(msg)
        self.assertFalse(return_msg.ok)
        self.assertIn("URLDownloadError", return_msg.error_message)
        self.assertIn(bad_url, return_msg.error_message)
        self.assertEqual(type(return_msg), JobReturnMsg)

    def test_img_classify_classifier_key(self):

        msg = JobMsg(task_name='classify_image',
                     tasks=[ClassifyImageMsg(
                         job_token='my_job',
                         image_loc=DataLocation(storage_type='url',
                                                key=TEST_URL),
                         extractor=DummyExtractor(),
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
                             trainer='minibatch',
                             nbr_epochs=1,
                             clf_type=clf_type,
                             labels=TrainingTaskLabels.example(),
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

    def test_multiple_feature_extract(self):
        extract_msg = ExtractFeaturesMsg(
            job_token='test_job',
            extractor=DummyExtractor(),
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
            extractor=DummyExtractor(),
            rowcols=[(1, 1)],
            image_loc=DataLocation(storage_type='url',
                                   key=TEST_URL),
            feature_loc=DataLocation(storage_type='memory',
                                     key='my_feats'))

        fail_extract_msg = ExtractFeaturesMsg(
            job_token='test_job',
            extractor=DummyExtractor(),
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
