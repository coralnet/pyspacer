import unittest
import warnings
import os

from spacer import tasks

from spacer.messages import DeployMsg


class TestDeploy(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)

    def tearDown(self):
        os.remove('baboon.png')

    def test_deploy_simple(self):
        msg = DeployMsg(
            pk=0,
            im_url='https://homepages.cae.wisc.edu/~ece533/images/baboon.png',
            feature_extractor_name='dummy',
            rowcols=[(100, 100), (200, 200)],
            classifier_key='legacy.model',
            bucketname='spacer-test',
        )

        results = tasks.deploy(msg)
        self.assertEqual(len(results.scores), len(msg.rowcols))


if __name__ == '__main__':
    unittest.main()
