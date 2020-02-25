import json
import os
import unittest

from spacer import config
from spacer.data_classes import \
    PointFeatures, \
    ImageFeatures, \
    ImageLabels, \
    ValResults


class TestDataClass(unittest.TestCase):
    """ Testing the default implementations of DataClass"""

    def test_repr(self):
        pf = PointFeatures.example()
        self.assertEqual(str(pf),
                         "{'col': 100, 'data': [1.1, 1.3, 1.12], 'row': 100}")


class TestPointFeatures(unittest.TestCase):

    def test_serialize(self):

        msg = PointFeatures.example()
        self.assertEqual(msg, PointFeatures.deserialize(
            msg.serialize()))
        self.assertEqual(msg, PointFeatures.deserialize(
            json.loads(json.dumps(msg.serialize()))))


class TestImageFeatures(unittest.TestCase):

    def test_serialize(self):

        msg = ImageFeatures.example()
        self.assertEqual(msg, ImageFeatures.deserialize(
            msg.serialize()))
        self.assertEqual(msg, ImageFeatures.deserialize(
            json.loads(json.dumps(msg.serialize()))))

    def test_legacy(self):
        """
        Loads a legacy feature file and make sure it's parsed correctly.
        """
        with open(os.path.join(config.LOCAL_FIXTURE_DIR,
                               'legacy.jpg.feats')) as fp:
            msg = ImageFeatures.deserialize(json.load(fp))
        self.assertEqual(msg.valid_rowcol, False)
        self.assertEqual(ImageFeatures.deserialize(
            msg.serialize()).valid_rowcol, False)

        self.assertTrue(isinstance(msg.point_features[0], PointFeatures))
        self.assertEqual(msg.npoints, len(msg.point_features))
        self.assertEqual(msg.feature_dim, len(msg.point_features[0].data))

        self.assertEqual(msg, ImageFeatures.deserialize(msg.serialize()))
        self.assertEqual(msg, ImageFeatures.deserialize(json.loads(
            json.dumps(msg.serialize()))))

    def test_getitem(self):
        msg = ImageFeatures.example()
        point_features = msg[(100, 100)]
        self.assertEqual(point_features[0], 1.1)

    def test_legacy_getitme(self):
        msg = ImageFeatures.example()
        msg.valid_rowcol = False
        self.assertRaises(ValueError, msg.__getitem__, (100, 100))


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