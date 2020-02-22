import itertools
import json
import unittest
from typing import Tuple

import numpy as np

from spacer import config
from spacer.messages import \
    ImageLabels, \
    ImageFeatures, \
    PointFeatures, \
    TrainClassifierMsg
from spacer.storage import storage_factory
from spacer.train_classifier import calc_batch_size, chunkify, \
    load_image_data, load_batch_data, calc_acc, trainer_factory


class TestCalcBatchSize(unittest.TestCase):

    def test1(self):

        images_per_batch, batches_per_epoch = calc_batch_size(1000, 10)
        self.assertEqual(images_per_batch, 10)
        self.assertEqual(batches_per_epoch, 1)

    def test2(self):
        images_per_batch, batches_per_epoch = calc_batch_size(3, 5)
        self.assertEqual(images_per_batch, 3)
        self.assertEqual(batches_per_epoch, 2)

    def test3(self):
        images_per_batch, batches_per_epoch = calc_batch_size(1, 5)
        self.assertEqual(images_per_batch, 1)
        self.assertEqual(batches_per_epoch, 5)


class TestChunkify(unittest.TestCase):

    def test1(self):
        out = chunkify(list(range(10)), 3)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[0]), 4)
        self.assertEqual(len(list(itertools.chain.from_iterable(out))), 10)

    def test2(self):
        out = chunkify(list(range(9)), 3)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[0]), 3)
        self.assertEqual(len(list(itertools.chain.from_iterable(out))), 9)

    def test3(self):
        out = chunkify(list(range(10)), 10)
        self.assertEqual(len(out), 10)
        self.assertEqual(len(out[0]), 1)
        self.assertEqual(len(list(itertools.chain.from_iterable(out))), 10)


class TestAcc(unittest.TestCase):

    def test_simple(self):
        self.assertEqual(calc_acc([1, 2, 3, 11], [1, 2, 1, 4]), 0.5)
        self.assertRaises(TypeError, calc_acc, [], [])
        self.assertRaises(ValueError, calc_acc, [1], [2, 1])
        self.assertRaises(TypeError, calc_acc, [1.1], [1])
        self.assertRaises(TypeError, calc_acc, [1], [1.1])


class TestLoadImageData(unittest.TestCase):

    def setUp(self):

        self.feat_key = 'tmp_features'
        self.storage = storage_factory('memory', '')

    def fixtures(self, in_order=True, valid_rowcol=True) \
            -> Tuple[ImageLabels, ImageFeatures]:

        labels = ImageLabels(
            data={self.feat_key: [(100, 100, 1),
                                  (200, 200, 2),
                                  (300, 300, 1)]})

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

        self.storage.store_string(self.feat_key,
                                  json.dumps(features.serialize()))

        return labels, features

    def test_simple(self):

        labels, features = self.fixtures(in_order=True)

        x, y = load_image_data(labels, self.feat_key, [1, 2], self.storage)

        self.assertEqual(y, [1, 2, 1])
        self.assertEqual(x[0], features.point_features[0].data)

    def test_reverse(self):
        """ here the order of features and labels are reversed.
        But it all still works thanks to row, col matching. """

        labels, features = self.fixtures(in_order=False)

        x, y = load_image_data(labels, self.feat_key, [1, 2], self.storage)

        self.assertEqual(y, [1, 2, 1])
        # Since the order is reversed, the first feature should be the second
        # vector of point_features list.
        self.assertEqual(x[0], features.point_features[1].data)

    def test_legacy_reverse(self):
        """
        Here we pretend the features are legacy such that row, col
        information is not available.
        """
        labels, features = self.fixtures(in_order=False, valid_rowcol=False)

        x, y = load_image_data(labels, self.feat_key, [1, 2], self.storage)

        self.assertEqual(y, [1, 2, 1])
        # Since the order is reversed, the first feature should be the second
        # vector of point_features list. But it is not.
        self.assertNotEqual(x[0], features.point_features[1].data)

    def test_smaller_labelset(self):
        """ Here we use a smaller labelset and assert
        that the right feature vector is kept """

        labels, features = self.fixtures(in_order=True, valid_rowcol=True)

        x, y = load_image_data(labels, self.feat_key, [1], self.storage)

        self.assertEqual(y, [1, 1])
        self.assertEqual(x[0], features.point_features[0].data)
        self.assertEqual(x[1], features.point_features[2].data)

    def test_other_small_labelset(self):
        """ Here we use a smaller labelset and assert
        that the right feature vector is kept """

        labels, features = self.fixtures(in_order=True, valid_rowcol=True)

        x, y = load_image_data(labels, self.feat_key, [2], self.storage)

        self.assertEqual(y, [2])
        self.assertEqual(x[0], features.point_features[1].data)

    def test_legacy_smaller_labelset(self):
        """
        Here we pretend the features are legacy such that row, col
        information is not available.
        """
        labels, features = self.fixtures(in_order=True, valid_rowcol=True)

        x, y = load_image_data(labels, self.feat_key, [1], self.storage)

        self.assertEqual(y, [1, 1])
        # Since the order is reversed, the first feature should be the second
        # vector of point_features list. But it is not.
        self.assertEqual(x[0], features.point_features[0].data)
        self.assertEqual(x[1], features.point_features[2].data)


class TestLoadBatchData(unittest.TestCase):

    def setUp(self):

        self.feat_key1 = 'tmp_features1'
        self.feat_key2 = 'tmp_features2'
        self.storage = storage_factory('memory', '')

    def fixtures(self, valid_rowcol=True) \
            -> Tuple[ImageLabels, ImageFeatures, ImageFeatures]:

        labels = ImageLabels(
            data={self.feat_key1: [(100, 100, 1),
                                   (200, 200, 2),
                                   (300, 300, 1)],
                  self.feat_key2: [(10, 10, 1),
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

        self.storage.store_string(self.feat_key1,
                                  json.dumps(features1.serialize()))

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

        self.storage.store_string(self.feat_key2,
                                  json.dumps(features2.serialize()))

        return labels, features1, features2

    def test_simple(self):

        labels, features1, features2 = self.fixtures(valid_rowcol=True)
        x, y = load_batch_data(labels, [self.feat_key1, self.feat_key2],
                               [1, 2], self.storage)

        self.assertEqual(x[0], features1.point_features[0].data)
        self.assertEqual(x[3], features2.point_features[0].data)

    def test_reverse_imkey_order(self):
        labels, features1, features2 = self.fixtures(valid_rowcol=True)
        x, y = load_batch_data(labels, [self.feat_key2, self.feat_key1],
                               [1, 2], self.storage)

        self.assertEqual(x[0], features2.point_features[0].data)
        self.assertEqual(x[3], features1.point_features[0].data)

    def test_one_label(self):

        labels, features1, features2 = self.fixtures(valid_rowcol=True)
        x, y = load_batch_data(labels, [self.feat_key2, self.feat_key1],
                               [1], self.storage)

        # Both images have two points with label=1
        self.assertEqual(len(x), 4)
        self.assertEqual(len(y), 4)


@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestDefaultTrainerDummyData(unittest.TestCase):

    def setUp(self):

        self.n_valdata = 20
        self.n_traindata = 200
        self.n_points = 10
        self.feature_dim = 5
        self.storage = storage_factory('memory', '')

    def make_image_features(self):

        pfs = [PointFeatures(row=itt,
                             col=itt,
                             data=list(np.random.multivariate_normal(
                                 np.ones(self.feature_dim)*itt % 2,
                                 np.eye(self.feature_dim))))
               for itt in range(self.n_points)]

        return ImageFeatures(point_features=pfs,
                             valid_rowcol=True,
                             feature_dim=self.feature_dim,
                             npoints=self.n_points)

    def make_labels(self, count):

        labels = ImageLabels(data={})
        for i in range(count):
            feats = self.make_image_features()
            imkey = 'val_{}'.format(i)
            self.storage.store_string(imkey, json.dumps(feats.serialize()))
            labels.data[imkey] = [
                (pf.row, pf.col, itt % 2) for itt, pf in
                enumerate(feats.point_features)
            ]
        return labels

    def test_gaussian(self):

        np.random.seed(0)
        labels = self.make_labels(self.n_valdata)
        self.storage.store_string('valdata', json.dumps(labels.serialize()))

        labels = self.make_labels(self.n_traindata)
        self.storage.store_string('traindata', json.dumps(labels.serialize()))

        msg = TrainClassifierMsg(
            pk=1,
            model_key='dummy',
            traindata_key='traindata',
            valdata_key='valdata',
            valresult_key='dummy',
            nbr_epochs=5,
            pc_models_key=[],
            pc_pks=[],
            bucketname='',
            storage_type='local'
        )

        trainer = trainer_factory(msg, self.storage)
        clf, val_results, return_message = trainer()

        # The way we rendered the data, accuracy is usually around 90%.
        # Adding some margin to account for randomness.
        # TODO: fix random seed; somehow the set above didn't work.
        self.assertGreater(return_message.acc,
                           0.80,
                           "Failure may be due to random generated numbers,"
                           "re-run tests.")

if __name__ == '__main__':
    unittest.main()
