import unittest
import sys
import json
import os
import boto

from boto.sqs.message import Message

sys.path.append('/root/caffe')
from spacer import mailman



class TestMailman(unittest.TestCase):
    
    def test_handle_message_input_type(self):
        self.assertRaises(TypeError, mailman.handle_message, 'sdf')
        self.assertRaises(KeyError, mailman.handle_message, {'not-task':'dummy'})
        self.assertRaises(ValueError, mailman.handle_message, {'task':'invalid-task'})


    def test_tasknames(self):
        m = self.makemessage()
        valid_tasks = ['extract_features', 'train_robot', 'classify_image']
        for task in valid_tasks:
            m['task'] = task
            status, res_body = mailman.handle_message(m)
            self.assertTrue(status)


    def test_sqsread(self):
        self.sendmessage(self.makemessage())
        mailman.grab_message()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def makemessage(self):
        m = {
        'task':'extract_features',
        'payload':{}
        }
        return m

    def sendmessage(self, body):
        queue = boto.sqs.connect_to_region("us-west-2").get_queue('spacer_jobs')
        m = Message()
        m.set_body(json.dumps(body))
        queue.write(m)


if __name__ == '__main__':

    # SETUP CREDENTIALS
    with open('../secrets.json') as data_file:    
        data = json.load(data_file)
        os.environ['AWS_ACCESS_KEY_ID'] = data['aws_access_key_id']
        os.environ['AWS_SECRET_ACCESS_KEY'] = data['aws_secret_access_key']

    suite = unittest.TestLoader().loadTestsFromTestCase(TestMailman)
    unittest.TextTestRunner(verbosity=2).run(suite)

    
    
    
    
