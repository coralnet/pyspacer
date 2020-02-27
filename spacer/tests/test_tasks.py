import os
import unittest
import warnings

from PIL import Image

from spacer import config
from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    DeployMsg, \
    DeployReturnMsg
from spacer.tasks import extract_features, train_classifier, deploy


class TestExtractFeatures(unittest.TestCase):

    def setUp(self):

        self.tmps = {
            'in': 'my_img.jpg',
            'out': 'my_output.json'
        }
        im = Image.new('RGB', (100, 100))
        im.save(self.tmps['in'])

    def tearDown(self):

        for tmp in self.tmps.values():
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_nominal(self):

        msg = ExtractFeaturesMsg(
            pk=0,
            feature_extractor_name='dummy',
            bucketname='',
            imkey=self.tmps['in'],
            rowcols=[(1, 1), (2, 2)],
            outputkey=self.tmps['out'],
            storage_type='filesystem'
        )
        return_msg = extract_features(msg)
        self.assertTrue(type(return_msg) == ExtractFeaturesReturnMsg)
        self.assertTrue(os.path.exists(self.tmps['out']))


class TestTrainClassifier(unittest.TestCase):

    def test_nominal(self):

        msg = TrainClassifierMsg(
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
        )
        return_msg = train_classifier(msg)
        self.assertTrue(type(return_msg) == TrainClassifierReturnMsg)


class TestDeploy(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)

    def tearDown(self):
        if os.path.exists('baboon.png'):
            os.remove('baboon.png')

    @unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to tests')
    def test_deploy_simple(self):
        msg = DeployMsg(
            pk=0,
            im_url='https://homepages.cae.wisc.edu/~ece533/images/baboon.png',
            feature_extractor_name='dummy',
            rowcols=[(100, 100), (200, 200)],
            classifier_key='legacy.model',
            bucketname='spacer-test',
        )

        return_msg = deploy(msg)
        self.assertEqual(len(return_msg.scores), len(msg.rowcols))
        self.assertTrue(type(return_msg.scores), DeployReturnMsg)


if __name__ == '__main__':
    unittest.main()
