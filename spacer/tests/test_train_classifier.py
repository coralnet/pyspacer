import random
import unittest
from unittest import mock

import numpy as np

from spacer import config
from spacer.data_classes import DataLocation, ImageLabels
from spacer.train_classifier import (
    ClassifierTrainer, MiniBatchTrainer, trainer_factory,
)
from spacer.train_utils import make_random_data
from spacer.tests.utils import spy_decorator


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
            pc_clf1, _ = MiniBatchTrainer()._train(
                train_data, ref_data, 1, clf_type)
            pc_clf2, _ = MiniBatchTrainer()._train(
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

    def test_serialize_default_only_class_path(self):
        trainer = MiniBatchTrainer()
        self.assertEqual(
            trainer.serialize(),
            {'class_path': 'spacer.train_classifier.MiniBatchTrainer'},
        )

    def test_serialize_with_custom_params(self):
        trainer = MiniBatchTrainer(batch_size=10000)
        data = trainer.serialize()
        self.assertIn('batch_size', data)
        self.assertEqual(data['batch_size'], 10000)

    def test_round_trip_with_custom_params(self):
        trainer = MiniBatchTrainer(batch_size=10000, sgd_loss='hinge')
        data = trainer.serialize()
        restored = ClassifierTrainer.deserialize(data)
        self.assertEqual(trainer, restored)
        self.assertEqual(restored.batch_size, 10000)
        self.assertEqual(restored.sgd_loss, 'hinge')


class TestTrain(unittest.TestCase):

    def do_basic_run(self, class_list, clf_type):

        n_traindata = 5
        n_refdata = 1
        points_per_image = 20
        feature_dim = 5
        num_epochs = 4
        feature_loc = DataLocation(storage_type='memory', key='')

        train_labels = make_random_data(
            n_traindata, class_list, points_per_image,
            feature_dim, feature_loc,
        )
        ref_labels = make_random_data(
            n_refdata, class_list, points_per_image,
            feature_dim, feature_loc,
        )

        clf_calibrated, ref_acc = MiniBatchTrainer()._train(
            train_labels, ref_labels,
            num_epochs, clf_type,
        )

        self.assertEqual(
            len(ref_acc), num_epochs,
            msg="Sanity check: expecting one ref_acc element per epoch")

    def test_lr_int_labels(self):
        self.do_basic_run([1, 2], 'LR')

    def test_mlp_int_labels(self):
        self.do_basic_run([1, 2], 'MLP')

    def test_lr_str_labels(self):
        self.do_basic_run(['Porites', 'CCA', 'Sand'], 'LR')

    def test_mlp_str_labels(self):
        self.do_basic_run(['Porites', 'CCA', 'Sand'], 'MLP')

    def test_mlp_annotation_threshold(self):
        """
        Test the default trainer's logic for automatically selecting MLP
        parameters based on an annotation-count threshold.
        """
        param_sets = [
            (11, 20, (100,), 1e-3),
            # 100*1000 = 100000 total annotations; threshold is 50000
            (100, 1000, (200, 100), 1e-4),
        ]

        for (n_traindata, points_per_image, hls, lr) in param_sets:
            feature_dim = 5
            class_list = [1, 2]
            num_epochs = 4
            feature_loc = DataLocation(storage_type='memory', key='')

            train_labels = make_random_data(
                n_traindata, class_list, points_per_image,
                feature_dim, feature_loc,
            )
            ref_labels = make_random_data(
                1, class_list, points_per_image,
                feature_dim, feature_loc,
            )

            clf_calibrated, ref_acc = MiniBatchTrainer()._train(
                train_labels, ref_labels, num_epochs, 'MLP')
            clf_param = clf_calibrated.get_params()['estimator']
            self.assertEqual(
                clf_param.hidden_layer_sizes, hls,
                msg="Hidden layer sizes should match the expected values"
                    " for the annotation count")
            self.assertEqual(
                clf_param.learning_rate_init, lr,
                msg="Learning rate init value should match the expected value"
                    " for the annotation count")

    def test_custom_batch_size(self):
        feature_loc = DataLocation(storage_type='memory', key='')
        train_labels = make_random_data(
            5, [1, 2], 20, 5, feature_loc)
        ref_labels = make_random_data(
            1, [1, 2], 20, 5, feature_loc)
        trainer = MiniBatchTrainer(batch_size=50)

        load_spied = spy_decorator(ImageLabels.load_data_in_batches)
        with mock.patch.object(
            ImageLabels, 'load_data_in_batches', load_spied
        ):
            clf, ref_acc = trainer._train(
                train_labels, ref_labels, 2, 'LR')

        self.assertEqual(len(ref_acc), 2)
        # Verify batch_size=50 was passed through to load_data_in_batches.
        # Called once per epoch with random_seed=epoch_number, plus once
        # for ref data loading via load_all_data().
        load_spied.mock_obj.assert_any_call(batch_size=50, random_seed=0)
        load_spied.mock_obj.assert_any_call(batch_size=50, random_seed=1)
        load_spied.mock_obj.assert_any_call(
            batch_size=None, random_seed=None)
        self.assertEqual(load_spied.mock_obj.call_count, 3)

    def test_explicit_mlp_params(self):
        feature_loc = DataLocation(storage_type='memory', key='')
        train_labels = make_random_data(
            5, [1, 2], 20, 5, feature_loc)
        ref_labels = make_random_data(
            1, [1, 2], 20, 5, feature_loc)
        trainer = MiniBatchTrainer(
            mlp_hidden_layer_sizes=(50,),
            mlp_learning_rate_init=0.01,
        )
        clf, ref_acc = trainer._train(train_labels, ref_labels, 2, 'MLP')
        clf_param = clf.get_params()['estimator']
        self.assertEqual(clf_param.hidden_layer_sizes, (50,))
        self.assertEqual(clf_param.learning_rate_init, 0.01)


if __name__ == '__main__':
    unittest.main()
