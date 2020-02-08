import unittest

from spacer import tasks
from spacer.messages import ExtractFeaturesMsg, TaskMsg
from spacer.mailman import handle_message

class TestDeploy(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_deploy_simple(self):

        payload = {
            'bucketname': 'coralnet-beijbom-dev',
            'im_url': 'https://coralnet-beijbom-dev.s3-us-west-2.amazonaws.com/media/images/04yv0o1o88.jpg',
            'modelname': 'vgg16_coralnet_ver1',
            'rowcols': [[100, 100], [200, 200]],
            'model': 'media/classifiers/15.model',
        }

        results = tasks.deploy(payload)
        self.assertEqual(results['ok'], 1)

    def test_deploy_error(self):

        payload = {
            'bucketname': 'coralnet-beijbom-dev',
            'im_url': 'https://nothing_here.jpg',
            'modelname': 'vgg16_coralnet_ver1',
            'rowcols': [[100, 100], [200, 200]],
            'model': 'media/classifiers/15.model',
        }

        results = tasks.deploy(payload)
        self.assertEqual(results['ok'], 0)


class TestExtractFeatures(unittest.TestCase):

    def test_caffe_extract(self):

        msg = ExtractFeaturesMsg(
            pk=1,
            modelname='vgg16_coralnet_ver1',
            bucketname='spacer-test',
            storage_type='s3',
            imkey='edinburgh3.jpg',
            rowcols=[(100, 100)],
            outputkey='edinburgh3.jpg.feats'
        )

        return_message = handle_message(TaskMsg('extract_features', msg))
        print(return_message)


if __name__ == '__main__':
    unittest.main()
