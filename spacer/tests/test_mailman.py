import json
import os
import unittest
import warnings
import time

from boto import sqs

from spacer.mailman import process_task, sqs_mailman
from spacer.messages import \
    TaskMsg, \
    TaskReturnMsg, \
    ExtractFeaturesMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    DeployMsg


class TestProcessFeatureExtract(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.url = 'https://homepages.cae.wisc.edu/~ece533/images/baboon.png'

    def tearDown(self):
        if os.path.exists('baboon.png'):
            os.remove('baboon.png')

    def test_input_type(self):
        self.assertRaises(AssertionError, process_task, 'sdf')

    def test_task_name(self):
        msg = TaskMsg.example()
        msg.task = 'invalid'
        self.assertRaises(AssertionError, process_task, msg)

    def test_failed_feature_extract(self):
        msg = TaskMsg(task='extract_features',
                      payload=ExtractFeaturesMsg(
                          pk=0,
                          feature_extractor_name='dummy',
                          bucketname='',
                          imkey='image_that_is_missing',
                          rowcols=[(1, 1)],
                          outputkey='does not matter',
                          storage_type='memory'
                      ))
        return_msg = process_task(msg)
        self.assertFalse(return_msg.ok)
        self.assertEqual(return_msg.error_message,
                         "KeyError('image_that_is_missing',)")
        self.assertTrue(type(return_msg), TaskReturnMsg)

    def test_failed_deploy_url(self):

        msg = TaskMsg(task='deploy',
                      payload=DeployMsg(
                          pk=0,
                          im_url='http://invalid_url.com',
                          feature_extractor_name='invalid_name',
                          rowcols=[(1, 1)],
                          classifier_key='nothing_here',
                          bucketname='',
                          storage_type='memory')
                      )
        return_msg = process_task(msg)
        self.assertFalse(return_msg.ok)
        self.assertEqual(return_msg.error_message,
                         "URLError(gaierror(8, 'nodename nor servname "
                         "provided, or ""not known'),)")
        self.assertTrue(type(return_msg), TaskReturnMsg)

    def test_failed_deploy_feature_extractor_name(self):

        msg = TaskMsg(task='deploy',
                      payload=DeployMsg(
                          pk=0,
                          im_url=self.url,
                          feature_extractor_name='invalid_name',
                          rowcols=[(1, 1)],
                          classifier_key='nothing_here',
                          bucketname='',
                          storage_type='memory')
                      )
        return_msg = process_task(msg)
        self.assertFalse(return_msg.ok)
        self.assertEqual(return_msg.error_message,
                         "AssertionError('Model name invalid_name "
                         "not registered',)")
        self.assertTrue(type(return_msg), TaskReturnMsg)

    def test_failed_deploy_classifier_key(self):

        msg = TaskMsg(task='deploy',
                      payload=DeployMsg(
                          pk=0,
                          im_url=self.url,
                          feature_extractor_name='dummy',
                          rowcols=[(1, 1)],
                          classifier_key='no_classifier_here',
                          bucketname='',
                          storage_type='memory')
                      )
        return_msg = process_task(msg)
        self.assertFalse(return_msg.ok)
        self.assertEqual(return_msg.error_message,
                         "KeyError('no_classifier_here',)")
        self.assertTrue(type(return_msg), TaskReturnMsg)

    def test_failed_train_classifier(self):

        msg = TaskMsg(task='train_classifier',
                      payload=TrainClassifierMsg(
                          pk=1,
                          model_key='my_trained_model',
                          trainer_name='minibatch',
                          traindata_key='my_traindata',
                          valdata_key='my_valdata',
                          valresult_key='my_valresults',
                          nbr_epochs=5,
                          pc_models_key=['my_previous_model1',
                                         'my_previous_model2',
                                         'my_pÍrevious_model3'],
                          pc_pks=[1, 2, 3],
                          bucketname='spacer-test',
                          storage_type='memory')
                      )
        return_msg = process_task(msg)
        self.assertFalse(return_msg.ok)
        self.assertEqual(return_msg.error_message,
                         "KeyError('my_traindata',)")
        self.assertTrue(type(return_msg), TaskReturnMsg)


class TestGrabMessage(unittest.TestCase):

    def setUp(self):
        self.queue_group = 'spacer_test'
        conn = sqs.connect_to_region("us-west-2")
        self.jobqueue = conn.get_queue('{}_jobs'.format(self.queue_group))
        if self.jobqueue is None:
            self.sqs_access = False
        else:
            self.sqs_access = True
        self.resqueue = conn.get_queue('{}_results'.format(self.queue_group))
        # Manually purge, since queue.purge is only allowed every 60 seconds.
        m = self.jobqueue.read()
        while m is not None:
            self.jobqueue.delete_message(m)
            m = self.jobqueue.read()
            print("Purged old message from jobqueue.")

        m = self.resqueue.read()
        while m is not None:
            self.resqueue.delete_message(m)
            m = self.resqueue.read()
            print("Purged old message from resqueue.")

    def post_job_get_result(self):
        found_message = False
        while not found_message:
            found_message = sqs_mailman(queue_group=self.queue_group)
            time.sleep(1)

        m_result = self.resqueue.read()
        while m_result is None:
            print('-> Mo message in result queue, trying again.')
            time.sleep(1)
            m_result = self.resqueue.read()
        return m_result

    def test_nonsense_body(self):
        if not self.sqs_access:
            return 1

        m_job = self.jobqueue.new_message(body=json.dumps(
            {'nonsense': 'value'}))
        self.jobqueue.write(m_job)

        m_result = self.post_job_get_result()

        body = json.loads(m_result.get_body())

        self.assertFalse(body['ok'])
        self.assertEqual(body['error_message'],
                         "Error deserializing message: KeyError('task',)")

    def test_valid_body_but_failed_process(self):

        if not self.sqs_access:
            return 1

        m_job = self.jobqueue.new_message(body=json.dumps(
            TaskMsg.example().serialize()))
        self.jobqueue.write(m_job)

        m_result = self.post_job_get_result()

        body = json.loads(m_result.get_body())
        return_msg = TaskReturnMsg.deserialize(body)
        self.assertFalse(return_msg.ok)
        self.assertEqual(return_msg.error_message,
                         'ValueError("unknown url '
                         'type: \'www.my.image.jpg\'",)')

    def test_nominal_train_classifier(self):

        msg = TaskMsg(
            task='train_classifier',
            payload = TrainClassifierMsg(
                pk=0,
                model_key='my_model',
                trainer_name='dummy',
                traindata_key='n/a',
                valdata_key='n/a',
                valresult_key='my_val_res',
                nbr_epochs=3,
                pc_models_key=[],
                pc_pks=[],
                bucketname='',
                storage_type='memory'
            ))
        m_job = self.jobqueue.new_message(body=json.dumps(msg.serialize()))
        self.jobqueue.write(m_job)

        m_result = self.post_job_get_result()

        body = json.loads(m_result.get_body())
        return_msg = TaskReturnMsg.deserialize(body)

        self.assertTrue(return_msg.ok)
        self.assertTrue(type(return_msg.results), TrainClassifierReturnMsg)




if __name__ == '__main__':
    unittest.main()
