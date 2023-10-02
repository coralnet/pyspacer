import json
import os
import unittest
import random

from spacer import config
from spacer.data_classes import \
    PointFeatures, \
    ImageFeatures, \
    ImageLabels, \
    ValResults
from spacer.messages import DataLocation
from .decorators import require_test_fixtures
from .utils import temp_filesystem_data_location


class TestDataClass(unittest.TestCase):
    """ Testing the default implementations of DataClass"""

    def test_repr(self):
        pf = PointFeatures.example()
        self.assertIn("'col': 100", str(pf))
        self.assertIn("'row': 100", str(pf))


class TestImageFeatures(unittest.TestCase):

    def test_legacy(self):
        """
        Loads a legacy feature file and make sure it's parsed correctly.
        """
        with open(os.path.join(config.LOCAL_FIXTURE_DIR,
                               'cnbeta.jpg.feats')) as fp:
            feats = ImageFeatures.deserialize(json.load(fp))
        self.assertEqual(feats.valid_rowcol, False)

        self.assertTrue(isinstance(feats.point_features[0], PointFeatures))
        self.assertEqual(feats.npoints, len(feats.point_features))
        self.assertEqual(feats.feature_dim, len(feats.point_features[0].data))

    @require_test_fixtures
    def test_legacy_from_s3(self):
        legacy_feat_loc = DataLocation(storage_type='s3',
                                       key='08bfc10v7t.png.featurevector',
                                       bucket_name=config.TEST_BUCKET)

        feats = ImageFeatures.load(legacy_feat_loc)
        self.assertEqual(feats.valid_rowcol, False)

        mem_loc = DataLocation(storage_type='memory', key='tmp')
        feats.store(mem_loc)
        reloaded_feats = ImageFeatures.load(mem_loc)
        self.assertEqual(feats, reloaded_feats)

    def test_getitem(self):
        msg = ImageFeatures.example()
        point_features = msg[(100, 100)]
        self.assertAlmostEqual(point_features[0], 1.1)

    def test_legacy_getitme(self):
        msg = ImageFeatures.example()
        msg.valid_rowcol = False
        self.assertRaises(ValueError, msg.__getitem__, (100, 100))


class TestImageFeaturesNumpyStore(unittest.TestCase):

    @require_test_fixtures
    def test_s3(self):
        s3_loc = DataLocation(
            storage_type='s3',
            key='tmp_feats',
            bucket_name=config.TEST_BUCKET
        )

        self._test_numpy_store(s3_loc)

        s3 = config.get_s3_conn()
        s3.Object(config.TEST_BUCKET, s3_loc.key).delete()

    def test_fs(self):
        with temp_filesystem_data_location() as fs_loc:
            self._test_numpy_store(fs_loc)

    def test_mem(self):
        mem_loc = DataLocation(
            storage_type='memory',
            key='tmp_feats'
        )
        self._test_numpy_store(mem_loc)

    def _test_numpy_store(self, feat_loc):

        n_pts = 200
        n_dim = 1280
        feats = ImageFeatures(
            point_features=[PointFeatures(row=i,
                                          col=i,
                                          data=[random.random()]*n_dim)
                            for i in range(n_pts)],
            valid_rowcol=True,
            feature_dim=n_dim,
            npoints=n_pts
        )
        with config.log_entry_and_exit('Storing feats vector'):
            feats.store(feat_loc)

        feats_reloaded = ImageFeatures.load(feat_loc)
        self.assertEqual(feats, feats_reloaded)

    def test_deprecated_serializer(self):
        feats = ImageFeatures.example()
        self.assertRaises(NotImplementedError, feats.serialize)


class TestFeatureLabels(unittest.TestCase):

    def test_serialize(self):

        msg = ImageLabels.example()
        self.assertEqual(msg, ImageLabels.deserialize(
            msg.serialize()))
        self.assertEqual(msg, ImageLabels.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_samples_per_image(self):
        msg = ImageLabels.example()
        self.assertEqual(msg.samples_per_image, 2)


class TestValResults(unittest.TestCase):

    def test_serialize(self):

        msg = ValResults.example()
        self.assertEqual(msg, ValResults.deserialize(
            msg.serialize()))
        self.assertEqual(msg, ValResults.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_asserts(self):

        gt = [0, 1, 2]
        est = [0, 1, 2]
        scores = [.2, .3, 5]
        classes = [123, 232, 11]

        try:
            ValResults(gt=gt,
                       est=est,
                       scores=scores,
                       classes=classes)
        except AssertionError:
            self.fail('AssertionError raised when it should not.')

        gt = [0, 1, 3]  # Index 3 is too large for 3 classes.
        est = [0, 1, 2]
        scores = [.2, .3, 5]
        classes = [123, 232, 11]

        self.assertRaises(AssertionError, ValResults,
                          gt=gt, est=est, scores=scores, classes=classes)

        gt = [0, 1, 2]
        est = [0, 1, 3]  # Index 3 is too large for 3 classes.
        scores = [.2, .3, 5]
        classes = [123, 232, 11]

        self.assertRaises(AssertionError, ValResults,
                          gt=gt, est=est, scores=scores, classes=classes)

        gt = [0, 1, 2]
        est = [0, 1, 2]
        scores = [.2, .3]  # Too few scores.
        classes = [123, 232, 11]

        self.assertRaises(AssertionError, ValResults,
                          gt=gt, est=est, scores=scores, classes=classes)

    @require_test_fixtures
    def test_legacy(self):
        legacy_loc = DataLocation(storage_type='s3',
                                  key='beta.valresult',
                                  bucket_name=config.TEST_BUCKET)

        res = ValResults.load(legacy_loc)
        self.assertEqual(res, ValResults.deserialize(json.loads(
            json.dumps(res.serialize()))))

if __name__ == '__main__':
    unittest.main()
