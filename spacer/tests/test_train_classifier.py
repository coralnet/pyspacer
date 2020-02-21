import unittest
import itertools
import json
from spacer.train_classifier import calc_batch_size, chunkify, \
    load_image_data, load_batch_data, calc_acc

from spacer.storage import storage_factory

from spacer.messages import FeatureLabels, ImageFeatures, PointFeatures


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

        self.tmp_json_file_name = 'tmp_data.json'
        self.storage = storage_factory('local', '')

    def tearDown(self):

        for tmp_file in [self.tmp_json_file_name]:
            self.storage.delete(tmp_file)

    def test_simple(self):

        labels = FeatureLabels(
            data={self.tmp_json_file_name: [(100, 100, 1), (200, 200, 2)]}
        )

        fv1 = [1.1, 1.2, 1.3]
        fv2 = [2.1, 2.2, 2.3]
        features = ImageFeatures(
            point_features=[
                PointFeatures(200, 200, fv2),
                PointFeatures(100, 100, fv1),
            ],
            valid_rowcol=True,
            feature_dim=3,
            npoints=2
        )

        self.storage.store_string(self.tmp_json_file_name,
                                  json.dumps(features.serialize()))

        x, y = load_image_data(labels, self.tmp_json_file_name, [1, 2], self.storage)

        self.assertEqual(y, [1, 2])
        self.assertEqual(x[0], fv1)







