from __future__ import annotations
import unittest

import numpy as np

from spacer import config
from spacer.data_classes import (
    DataLocation, ImageLabels, PointFeatures, ImageFeatures)
from spacer.exceptions import RowColumnMismatchError, RowColumnMissingError
from spacer.train_utils import (
    calc_acc,
    evaluate_classifier,
    make_random_data,
    train,
)


class TestMakeRandom(unittest.TestCase):

    def test_nominal(self):

        n_traindata = 20
        points_per_image = 200
        feature_dim = 5
        class_list = [1, 2]

        features_loc_template = DataLocation(storage_type='memory', key='')

        traindata = make_random_data(n_traindata,
                                     class_list,
                                     points_per_image,
                                     feature_dim,
                                     features_loc_template)

        self.assertEqual(
            len(traindata), n_traindata,
            msg="Length (image count) should be as expected")
        self.assertEqual(
            traindata.label_count, n_traindata*points_per_image,
            msg="Label count should be as expected")

        one_feature_loc = next(iter(traindata.keys()))
        feats = ImageFeatures.load(one_feature_loc)
        self.assertEqual(
            feats.feature_dim, feature_dim,
            msg="Should have created features as expected")


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

        clf_calibrated, ref_acc = train(
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

    def test_mlp_hybrid_mode(self):

        param_sets = [
            (11, 20, (100,), 1e-3),
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

            clf_calibrated, ref_acc = train(
                train_labels, ref_labels, num_epochs, 'MLP')
            clf_param = clf_calibrated.get_params()['estimator']
            self.assertEqual(
                clf_param.hidden_layer_sizes, hls,
                msg="Hidden layer sizes should correspond to label count")
            self.assertEqual(
                clf_param.learning_rate_init, lr,
                msg="Learning rate init value should correspond to label"
                    " count")


class TestEvaluateClassifier(unittest.TestCase):

    def test_simple(self):
        class_list = [1, 2]
        points_per_image = 4
        feature_dim = 5
        feature_loc = DataLocation(storage_type='memory', key='')
        train_data = make_random_data(
            9, class_list, points_per_image, feature_dim, feature_loc)
        ref_data = make_random_data(
            1, class_list, points_per_image, feature_dim, feature_loc)
        val_data = make_random_data(
            3, class_list, points_per_image, feature_dim, feature_loc)

        for clf_type in config.CLASSIFIER_TYPES:
            clf, _ = train(train_data, ref_data, 1, clf_type)

            gts, ests, scores = evaluate_classifier(clf, val_data)
            # Sanity checks
            self.assertTrue(all([gt in class_list for gt in gts]))
            self.assertTrue(all([est in class_list for est in ests]))
            self.assertTrue(all([0 < s < 1 for s in scores]))


class TestImageLabelsBatching(unittest.TestCase):
    """
    ImageLabels: batching logic.
    """
    @classmethod
    def setUpClass(cls):
        config.filter_warnings()

        cls.classes = [1, 2]
        cls.feature_dim = 5
        cls.feature_loc = DataLocation(storage_type='memory', key='')

    def test_below_batch_size(self):
        im_count = 5
        points_per_image = 10
        random_seed = 1
        image_labels = make_random_data(
            im_count, self.classes, points_per_image,
            self.feature_dim, self.feature_loc)
        batches = [
            batch for batch in image_labels.load_data_in_batches(
                # 5*10 < 1000
                batch_size=1000,
                random_seed=random_seed,
            )
        ]

        self.assertEqual(len(batches), 1, msg="Should have 1 batch")
        loaded_features, loaded_labels = batches[0]
        self.assertEqual(
            len(loaded_features), 50, msg="Should have 50 point features")
        self.assertEqual(
            len(loaded_labels), 50, msg="Should have 50 labels")

    def test_one_full_batch(self):
        im_count = 5
        points_per_image = 20
        random_seed = 1
        image_labels = make_random_data(
            im_count, self.classes, points_per_image,
            self.feature_dim, self.feature_loc)
        batches = [
            batch for batch in image_labels.load_data_in_batches(
                # 5*20 == 100
                batch_size=100,
                random_seed=random_seed,
            )
        ]

        self.assertEqual(len(batches), 1, msg="Should have 1 batch")
        loaded_features, loaded_labels = batches[0]
        self.assertEqual(
            len(loaded_features), 100, msg="Should have 100 point features")
        self.assertEqual(
            len(loaded_labels), 100, msg="Should have 100 labels")

    def test_no_batch_size(self):
        im_count = 5
        points_per_image = 20
        random_seed = 1
        image_labels = make_random_data(
            im_count, self.classes, points_per_image,
            self.feature_dim, self.feature_loc)
        # Everything is in one batch
        loaded_features, loaded_labels = image_labels.load_all_data(
            random_seed=random_seed)

        self.assertEqual(
            len(loaded_features), 100, msg="Should have 100 point features")
        self.assertEqual(
            len(loaded_labels), 100, msg="Should have 100 labels")

    def test_multiple_batches(self):
        im_count = 9
        points_per_image = 25
        random_seed = 1
        image_labels = make_random_data(
            im_count, self.classes, points_per_image,
            self.feature_dim, self.feature_loc)
        batches = [
            batch for batch in image_labels.load_data_in_batches(
                # 9*25 > 100
                batch_size=100,
                random_seed=random_seed,
            )
        ]

        self.assertEqual(len(batches), 3, msg="Should have 3 batches")

        batch_1_features, batch_1_labels = batches[0]
        self.assertEqual(
            len(batch_1_features), 100,
            msg="Batch 1 should have 100 point features")
        self.assertEqual(
            len(batch_1_labels), 100, msg="Batch 1 should have 100 labels")

        batch_2_features, batch_2_labels = batches[1]
        self.assertEqual(
            len(batch_2_features), 100,
            msg="Batch 2 should have 100 point features")
        self.assertEqual(
            len(batch_2_labels), 100, msg="Batch 2 should have 100 labels")

        batch_3_features, batch_3_labels = batches[2]
        self.assertEqual(
            len(batch_3_features), 25,
            msg="Batch 3 should have 25 point features")
        self.assertEqual(
            len(batch_3_labels), 25, msg="Batch 3 should have 25 labels")

    def test_one_image_split_between_batches(self):
        im_count = 1
        points_per_image = 101
        random_seed = 1
        image_labels = make_random_data(
            im_count, self.classes, points_per_image,
            self.feature_dim, self.feature_loc)
        batches = [
            batch for batch in image_labels.load_data_in_batches(
                # 101 > 100
                batch_size=100,
                random_seed=random_seed,
            )
        ]

        self.assertEqual(len(batches), 2, msg="Should have 2 batches")

        batch_1_features, batch_1_labels = batches[0]
        self.assertEqual(
            len(batch_1_features), 100,
            msg="Batch 1 should have 100 point features")
        self.assertEqual(
            len(batch_1_labels), 100, msg="Batch 1 should have 100 labels")

        batch_2_features, batch_2_labels = batches[1]
        self.assertEqual(
            len(batch_2_features), 1, msg="Batch 2 should have 1 point feature")
        self.assertEqual(
            len(batch_2_labels), 1, msg="Batch 2 should have 1 label")

    def test_repeatable_with_same_random_state(self):
        im_count = 20
        points_per_image = 1
        random_seed = 13
        image_labels = make_random_data(
            im_count, self.classes, points_per_image,
            self.feature_dim, self.feature_loc)

        batch_1_features, _ = image_labels.load_all_data(
            random_seed=random_seed)
        batch_2_features, _ = image_labels.load_all_data(
            random_seed=random_seed)

        # Should be pretty unlikely that two random-shuffled
        # 20-feature lists get the same ordering, unless they were
        # seeded the same.
        for i in range(im_count):
            self.assertTrue(
                np.array_equal(batch_1_features[i], batch_2_features[i]),
                msg=f"Element {i} in both results should be the same")


class TestImageLabelsLoading(unittest.TestCase):
    """
    ImageLabels: feature loading logic.
    """
    @classmethod
    def setUpClass(cls):
        config.filter_warnings()
        cls.feat1_loc = DataLocation(storage_type='memory',
                                     key='tmp_features1')
        cls.feat2_loc = DataLocation(storage_type='memory',
                                     key='tmp_features2')

    def fixtures(self, valid_rowcol=True) \
            -> tuple[ImageLabels, ImageFeatures, ImageFeatures]:

        labels = ImageLabels(
            data={self.feat1_loc: [(100, 100, 1),
                                   (200, 200, 2),
                                   (300, 300, 1)],
                  self.feat2_loc: [(10, 10, 1),
                                   (20, 20, 2),
                                   (30, 30, 1)]})

        features1 = ImageFeatures(
            point_features=[
                PointFeatures(100, 100, [1.1, 1.2, 1.3]),
                PointFeatures(200, 200, [2.1, 2.2, 2.3]),
                PointFeatures(300, 300, [3.1, 3.2, 3.3]),
            ],
            valid_rowcol=valid_rowcol,
            feature_dim=3,
            npoints=3
        )
        features1.store(self.feat1_loc)

        features2 = ImageFeatures(
            point_features=[
                PointFeatures(10, 10, [10.1, 10.2, 10.3]),
                PointFeatures(20, 20, [20.1, 20.2, 20.3]),
                PointFeatures(30, 30, [30.1, 30.2, 30.3]),
            ],
            valid_rowcol=valid_rowcol,
            feature_dim=3,
            npoints=3
        )
        features2.store(self.feat2_loc)

        return labels, features1, features2

    def test_simple(self):

        labels, features1, features2 = self.fixtures(valid_rowcol=True)
        loaded_features, _ = labels.load_all_data()

        for a, b in [
            (loaded_features[0], features1.point_features[0].data),
            (loaded_features[1], features1.point_features[1].data),
            (loaded_features[2], features1.point_features[2].data),
            (loaded_features[3], features2.point_features[0].data),
            (loaded_features[4], features2.point_features[1].data),
            (loaded_features[5], features2.point_features[2].data),
        ]:
            self.assertTrue(np.array_equal(a, b))


class TestImageLabelsLoading2(unittest.TestCase):
    """
    ImageLabels: feature loading logic, more tests.
    """
    @classmethod
    def setUpClass(cls):
        config.filter_warnings()
        cls.feat_key = 'tmp_features'
        cls.feature_loc = DataLocation(storage_type='memory',
                                       key=cls.feat_key)

    def fixtures(
        self,
        in_order=True,
        valid_rowcol=True,
        rowcol_mismatch=False,
    ) -> tuple[ImageLabels, ImageFeatures]:

        if rowcol_mismatch:
            annotations = [
                (100, 100, 1),
                (200, 200, 2),
                (300, 299, 1),
            ]
        else:
            annotations = [
                (100, 100, 1),
                (200, 200, 2),
                (300, 300, 1),
            ]
        image_labels = ImageLabels(data={self.feature_loc: annotations})

        fv1 = [1.1, 1.2, 1.3]
        fv2 = [2.1, 2.2, 2.3]
        fv3 = [3.1, 3.2, 3.3]
        if in_order:
            pfs = [
                PointFeatures(100, 100, fv1),
                PointFeatures(200, 200, fv2),
                PointFeatures(300, 300, fv3),
            ]
        else:
            # Position 1 and 2 switched compared to the labels.
            pfs = [
                PointFeatures(200, 200, fv2),
                PointFeatures(100, 100, fv1),
                PointFeatures(300, 300, fv3),
            ]

        features = ImageFeatures(
            point_features=pfs,
            valid_rowcol=valid_rowcol,
            feature_dim=3,
            npoints=3
        )
        features.store(self.feature_loc)

        return image_labels, features

    def test_simple(self):

        labels, features = self.fixtures(in_order=True)
        x, y = labels.load_all_data()

        self.assertEqual(list(y), [1, 2, 1])
        self.assertTrue(np.array_equal(x[0], features.point_features[0].data))

    def test_scrambled(self):
        """
        Here the feature ordering is scrambled.
        But the load result is still the same thanks to row, col matching.
        """
        labels, features = self.fixtures(in_order=False)
        x, y = labels.load_all_data()

        self.assertEqual(list(y), [1, 2, 1])
        # Same elements, different order relative to features.
        self.assertTrue(np.array_equal(x[0], features.point_features[1].data))
        self.assertTrue(np.array_equal(x[1], features.point_features[0].data))
        self.assertTrue(np.array_equal(x[2], features.point_features[2].data))

    def test_legacy(self):
        """
        Here we pretend the features are legacy such that row, col
        information is not available.
        """
        labels, features = self.fixtures(valid_rowcol=False)

        with self.assertRaises(RowColumnMissingError) as cm:
            labels.load_all_data()
        self.assertEqual(
            str(cm.exception),
            f"{self.feat_key}: Features without rowcols are no longer"
            f" supported for training.")

    def test_rowcol_mismatch(self):
        """
        Labels has a row-column pair that's not in features.
        """
        labels, features = self.fixtures(rowcol_mismatch=True)

        with self.assertRaises(RowColumnMismatchError) as cm:
            labels.load_all_data()
        self.assertEqual(
            str(cm.exception),
            f"{self.feat_key}: The labels' row-column positions don't match"
            f" those of the feature vector (example: (300, 299)).")


class TestAcc(unittest.TestCase):

    def test_simple(self):
        self.assertEqual(calc_acc([1, 2, 3, 11], [1, 2, 1, 4]), 0.5)
        self.assertRaises(ValueError, calc_acc, [], [])
        self.assertRaises(ValueError, calc_acc, [1], [2, 1])


if __name__ == '__main__':
    unittest.main()
