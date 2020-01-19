# import unittest
# import sys
# import json
# import boto
#
# from boto.sqs.message import Message
#
# sys.path.append('/root/caffe')
#
# from spacer import mailman  # noqa
#
#
# class TestMailman(unittest.TestCase):
#
#     @staticmethod
#     def sendmessage(body, queue):
#         queue = boto.sqs.connect_to_region("us-west-2").get_queue(queue)
#         m = Message()
#         m.set_body(json.dumps(body))
#         queue.write(m)
#
#     def test_handle_message_input_type(self):
#         self.assertRaises(TypeError, mailman.handle_message, 'sdf')
#         self.assertRaises(KeyError, mailman.handle_message, {'not-task': 'dummy'})
#         self.assertRaises(ValueError, mailman.handle_message, {'task': 'invalid-task'})
#
#     def test_extract_features_simple(self):
#         """
#         This particular image caused trouble on the production server.
#         The image file iteself is lightly corrupted, and PIL doesn't quite like it.
#         """
#         payload = {
#             'rowcols': [[100, 100], [200, 200]],
#             'modelname': 'vgg16_coralnet_ver1',
#             'outputkey': 'edinburgh3.jpg.feats',
#             'imkey': 'edinburgh3.jpg',
#             'bucketname': 'spacer-test',
#             'pk': 'dummy'
#         }
#         message = {
#             'task': 'extract_features',
#             'payload': payload
#
#         }
#         self.sendmessage(message, 'spacer_test_jobs')
#
#         result_message = mailman.handle_message(message)
#         self.assertTrue(('model_was_cashed' in result_message))
#
#     def test_cornercase1(self):
#         """
#         This particular image caused trouble on the production server.
#         The image file itself is lightly corrupted, and PIL doesn't quite like it.
#         """
#         payload = {
#             "rowcols": [[148, 50], [60, 425]],
#             "modelname": "vgg16_coralnet_ver1",
#             "outputkey": "kh6dydiix0.jpeg.feats",
#             "imkey": "kh6dydiix0.jpeg",
#             "bucketname": "spacer-test",
#             "pk": "kh6dydiix0"
#         }
#         message = {"task": "extract_features", 'payload': payload}
#
#         self.sendmessage(message, 'spacer_test_jobs')
#         result_message = mailman.handle_message(message)
#         self.assertTrue(('model_was_cashed' in result_message))
#
#     def test_cornercase2(self):
#         """
#         This particular image caused trouble on the production server.
#         The image file itself is lightly corrupted, and PIL doesn't quite like it.
#         """
#         message = {"task": "extract_features",
#                    "payload": {
#                        "rowcols": [[190, 226], [25, 359]],
#                        "modelname": "vgg16_coralnet_ver1",
#                        "outputkey": "sfq2mr5qbs.jpeg.feats",
#                        "imkey": "sfq2mr5qbs.jpeg",
#                        "bucketname": "spacer-test",
#                        "pk": "sfq2mr5qbs"}
#                    }
#         self.sendmessage(message, 'spacer_test_jobs')
#         result_message = mailman.handle_message(message)
#         self.assertTrue(('model_was_cashed' in result_message))
#
#     def setUp(self):
#         pass
#
#     def tearDown(self):
#         pass
#
#
# if __name__ == '__main__':
#
#     suite = unittest.TestLoader().loadTestsFromTestCase(TestMailman)
#     unittest.TextTestRunner(verbosity=2).run(suite)
#
