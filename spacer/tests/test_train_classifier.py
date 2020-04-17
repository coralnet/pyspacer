import random
import unittest

import numpy as np

from spacer import config
from spacer.train_classifier import trainer_factory
from spacer.train_utils import make_random_data, train
from spacer.messages import DataLocation


@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestDefaultTrainerDummyData(unittest.TestCase):

    def setUp(self):
        config.filter_warnings()
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
        feature_loc = DataLocation(storage_type='memory', key='')
        train_data = make_random_data(n_valdata,
                                      class_list,
                                      points_per_image,
                                      feature_dim,
                                      feature_loc)

        val_data = make_random_data(n_traindata,
                                    class_list,
                                    points_per_image,
                                    feature_dim,
                                    feature_loc)

        trainer = trainer_factory('minibatch')
        pc_clf1, _ = train(train_data, feature_loc, 1)
        pc_clf2, _ = train(train_data, feature_loc, 1)

        clf, val_results, return_message = trainer(train_data,
                                                   val_data,
                                                   num_epochs,
                                                   [pc_clf1, pc_clf2],
                                                   feature_loc)

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


if __name__ == '__main__':
    unittest.main()
