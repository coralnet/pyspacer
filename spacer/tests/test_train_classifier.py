import json
import random
import unittest
import warnings

import numpy as np

from spacer import config
from spacer.storage import storage_factory
from spacer.train_classifier import trainer_factory
from spacer.train_utils import make_random_data
from spacer.data_classes import DataLocation


@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestDefaultTrainerDummyData(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        np.random.seed(0)
        random.seed(0)

    def test_simple(self):
        n_valdata = 20
        n_traindata = 200
        points_per_image = 20
        feature_dim = 5
        class_list = [1, 2]
        num_epochs = 4

        # First create data to train on.
        features_loc_template = DataLocation(storage_type='memory', key='')
        traindata = make_random_data(n_valdata,
                                     class_list,
                                     points_per_image,
                                     feature_dim,
                                     features_loc_template)

        valdata = make_random_data(n_traindata,
                                   class_list,
                                   points_per_image,
                                   feature_dim,
                                   features_loc_template)

        # Then use the dummy-trainer to train two "previous" classifier
        trainer = trainer_factory('dummy',
                                  dummy_kwargs={'feature_dim': feature_dim})

        clf1, _, _ = trainer(None, None, 2, [], features_loc_template)
        clf2, _, _ = trainer(None, None, 2, [], features_loc_template)

        trainer = trainer_factory('minibatch')
        clf, val_results, return_message = trainer(traindata,
                                                   valdata,
                                                   num_epochs,
                                                   [clf1, clf2],
                                                   features_loc_template)

        # The way we rendered the data, accuracy is usually around 90%.
        # Adding some margin to account for randomness.
        # TODO: fix random seed; somehow the set above didn't work.
        # Seems it was a known problem in 0.17.1
        # (we are stuck at 0.17.1 due to caffe dependencies).
        # https://github.com/scikit-learn/scikit-learn/issues/10237
        self.assertGreater(return_message.acc,
                           0.75,
                           "Failure may be due to random generated numbers,"
                           "re-run tests.")
        self.assertEqual(len(return_message.pc_accs), 2)
        self.assertEqual(len(return_message.ref_accs), num_epochs)


class TestDummyTrainer(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)

    def test_simple(self):
        trainer = trainer_factory('dummy')
        features_loc_template = DataLocation(storage_type='memory', key='')
        clf, val_results, return_msg = trainer(None, None, 2, [],
                                               features_loc_template)

        self.assertGreater(return_msg.acc, 0.01,
                           "Failure may be due to random generated numbers,"
                           "re-run tests.")


if __name__ == '__main__':
    unittest.main()
