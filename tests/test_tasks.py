import unittest

from spacer import tasks


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
        print(results)

if __name__ == '__main__':
    unittest.main()
