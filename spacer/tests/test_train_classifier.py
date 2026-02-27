import random
import unittest

import numpy as np

from spacer import config
from spacer.data_classes import DataLocation
from spacer.train_classifier import (
    ClassifierTrainer, MiniBatchTrainer, trainer_factory,
)
from spacer.train_utils import make_random_data, train


class TestDefaultTrainerDummyData(unittest.TestCase):

    def setUp(self):
        config.filter_warnings()

        np.random.seed(0)
        random.seed(0)

    def test_simple(self):
        n_traindata = 160
        n_refdata = 20
        n_valdata = 20
        points_per_image = 20
        feature_dim = 5
        class_list = [1, 2]
        num_epochs = 4

        # First create data to train on.
        feature_loc = DataLocation(storage_type='memory', key='')
        train_data = make_random_data(
            n_traindata, class_list, points_per_image,
            feature_dim, feature_loc,
        )
        ref_data = make_random_data(
            n_refdata, class_list, points_per_image,
            feature_dim, feature_loc,
        )
        val_data = make_random_data(
            n_valdata, class_list, points_per_image,
            feature_dim, feature_loc,
        )

        trainer = trainer_factory('minibatch')
        for clf_type in config.CLASSIFIER_TYPES:
            # 2 previous classifiers
            pc_clf1, _ = train(
                train_data, ref_data, 1, clf_type)
            pc_clf2, _ = train(
                train_data, ref_data, 1, clf_type)

            clf, val_results, return_message = trainer(
                dict(train=train_data, ref=ref_data, val=val_data),
                num_epochs,
                [pc_clf1, pc_clf2],
                clf_type,
            )

            # The way we rendered the data, accuracy is usually around 90%.
            # Adding some margin to account for randomness. Due to randomizer
            # seeding, results should be the same when re-running in the same
            # environment, but can change when the environment changes.
            # Results also heavily depend on the implementation of
            # make_random_data().
            self.assertGreater(return_message.acc,
                               0.75,
                               "Failure may be due to random generated numbers."
                               " Consider lowering the acc threshold.")
            self.assertEqual(len(return_message.pc_accs), 2)
            self.assertEqual(len(return_message.ref_accs), num_epochs)


class TestTrainerFactory(unittest.TestCase):

    def test_builtin_alias(self):
        trainer = trainer_factory('minibatch')
        self.assertIsInstance(trainer, MiniBatchTrainer)

    def test_class_path(self):
        trainer = trainer_factory('spacer.train_classifier.MiniBatchTrainer')
        self.assertIsInstance(trainer, MiniBatchTrainer)

    def test_invalid_path(self):
        with self.assertRaises((ImportError, ModuleNotFoundError)):
            trainer_factory('no.such.module.Trainer')

    def test_invalid_class_name(self):
        with self.assertRaises(AttributeError):
            trainer_factory('spacer.train_classifier.NoSuchTrainer')

    def test_not_a_trainer(self):
        # ValResults is not a ClassifierTrainer subclass
        with self.assertRaises(TypeError):
            trainer_factory('spacer.data_classes.ValResults')


class TestTrainerSerialization(unittest.TestCase):

    def test_serialize_round_trip(self):
        trainer = MiniBatchTrainer()
        data = trainer.serialize()
        restored = ClassifierTrainer.deserialize(data)
        self.assertIsInstance(restored, MiniBatchTrainer)
        self.assertEqual(trainer, restored)

    def test_serialize_content(self):
        trainer = MiniBatchTrainer()
        data = trainer.serialize()
        self.assertEqual(
            data,
            {'class_path': 'spacer.train_classifier.MiniBatchTrainer'},
        )

    def test_repr(self):
        trainer = MiniBatchTrainer()
        self.assertIn('MiniBatchTrainer', repr(trainer))

    def test_equality(self):
        self.assertEqual(MiniBatchTrainer(), MiniBatchTrainer())


if __name__ == '__main__':
    unittest.main()
