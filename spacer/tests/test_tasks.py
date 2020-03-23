import os
import unittest
import warnings
from PIL import Image


from spacer.train_utils import make_random_data
from spacer import config
from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    ClassifyReturnMsg, \
    DataLocation

from spacer.tasks import extract_features, train_classifier
from spacer.train_utils import train
from spacer.storage import store_classifier


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
            job_token='test',
            feature_extractor_name='dummy',
            image_loc=DataLocation(storage_type='filesystem',
                                   key=self.tmps['in']),
            rowcols=[(1, 1), (2, 2)],
            feature_loc=DataLocation(storage_type='filesystem',
                                     key=self.tmps['out'])
        )
        return_msg = extract_features(msg)
        self.assertTrue(type(return_msg) == ExtractFeaturesReturnMsg)
        self.assertTrue(os.path.exists(self.tmps['out']))


class TestTrainClassifier(unittest.TestCase):

    def test_nominal(self):

        # Set some hyper parameters for data generation
        n_valdata = 20
        n_traindata = 200
        points_per_image = 20
        feature_dim = 5
        class_list = [1, 2]

        # Create train and val data.
        features_loc_template = DataLocation(storage_type='memory', key='')

        traindata_loc = DataLocation(storage_type='memory', key='traindata')
        traindata = make_random_data(n_valdata,
                                     class_list,
                                     points_per_image,
                                     feature_dim,
                                     features_loc_template)
        traindata.store(traindata_loc)

        valdata = make_random_data(n_traindata,
                                   class_list,
                                   points_per_image,
                                   feature_dim,
                                   features_loc_template)
        valdata_loc = DataLocation(storage_type='memory', key='traindata')
        valdata.store(valdata_loc)

        # Train once by calling directly so that we have a previous classifier.
        clf, _ = train(traindata, features_loc_template, 1)

        previous_classifier_loc = DataLocation(storage_type='memory',
                                               key='pc')
        store_classifier(previous_classifier_loc, clf)

        msg = TrainClassifierMsg(
            job_token='test',
            trainer_name='minibatch',
            nbr_epochs=1,
            traindata_loc=traindata_loc,
            valdata_loc=valdata_loc,
            features_loc=features_loc_template,
            previous_model_locs=[previous_classifier_loc],
            model_loc=DataLocation(storage_type='memory', key='model'),
            valresult_loc=DataLocation(storage_type='memory', key='val_res')
        )
        return_msg = train_classifier(msg)
        self.assertTrue(type(return_msg) == TrainClassifierReturnMsg)


@unittest.skip
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
        self.assertTrue(type(return_msg.scores), ClassifyReturnMsg)


if __name__ == '__main__':
    unittest.main()
