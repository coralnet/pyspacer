import json
import os
import unittest
import warnings
from io import BytesIO

import boto
from PIL import Image

from spacer import config
from spacer.storage import storage_factory
from spacer.messages import ImageFeatures

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier


class TestLocalStorage(unittest.TestCase):

    def setUp(self):
        self.tmp_image_file_name = 'tmp_image.jpg'
        self.tmp_json_file_name = 'tmp_data.json'
        self.tmp_model_file_name = 'tmp_model.pkl'
        self.storage = storage_factory('local', '')

    def tearDown(self):

        for tmp_file in [self.tmp_json_file_name,
                         self.tmp_image_file_name,
                         self.tmp_model_file_name]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    def test_load_image(self):

        img = Image.new('RGB', (100, 200))
        img.save(self.tmp_image_file_name)

        img2 = self.storage.load_image(self.tmp_image_file_name)

        self.assertEqual(img.size[0], img2.size[0])
        self.assertEqual(img.size[1], img2.size[1])

    def test_load_legacy_features(self):
        message_str = self.storage.load_string(os.path.join(
            config.LOCAL_FIXTURE_DIR, 'legacy.jpg.feats'))
        feats = ImageFeatures.deserialize(json.loads(message_str))
        self.assertTrue(isinstance(feats, ImageFeatures))
        self.assertFalse(feats.valid_rowcol)

    def test_string_store_load(self):

        data = json.dumps({'a': 1, 'b': 2})

        self.storage.store_string(self.tmp_json_file_name, data)
        data2 = self.storage.load_string(self.tmp_json_file_name)
        self.assertEqual(data, data2)

    def test_delete(self):

        data = json.dumps({'a': 1, 'b': 2})
        self.storage.store_string(self.tmp_json_file_name, data)
        self.storage.delete(self.tmp_json_file_name)
        self.assertFalse(os.path.exists(self.tmp_json_file_name))

    def test_load_legacy_model(self):
        clf = self.storage.load_classifier(os.path.join(
            config.LOCAL_FIXTURE_DIR, 'legacy.model'))
        self.assertTrue(isinstance(clf, CalibratedClassifierCV))

    def test_load_store_model(self):
        clf = CalibratedClassifierCV(SGDClassifier())
        self.storage.store_classifier(self.tmp_model_file_name, clf)
        self.assertTrue(self.storage.exists(self.tmp_model_file_name))

        clf2 = self.storage.load_classifier(self.tmp_model_file_name)
        self.assertTrue(isinstance(clf2, CalibratedClassifierCV))


@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestS3Storage(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.tmp_image_key = 'tmp_image.jpg'
        self.tmp_json_key = 'tmp_data.json'
        self.tmp_model_key = 'tmp_model.pkl'
        self.storage = storage_factory('s3', 'spacer-test')

        conn = boto.connect_s3()
        self.bucket = conn.get_bucket('spacer-test')

    def tearDown(self):
        self.bucket.delete_key(self.tmp_image_key)
        self.bucket.delete_key(self.tmp_json_key)
        self.bucket.delete_key(self.tmp_model_key)

    def test_load_image(self):

        img = Image.new('RGB', (100, 200))

        with BytesIO() as stream:
            img.save(stream, 'JPEG')
            stream.seek(0)
            key = self.bucket.new_key(self.tmp_image_key)
            key.set_contents_from_file(stream)

        img2 = self.storage.load_image(self.tmp_image_key)

        self.assertEqual(img.size[0], img2.size[0])
        self.assertEqual(img.size[1], img2.size[1])

    def test_load_legacy_features(self):
        message_str = self.storage.load_string('legacy.jpg.feats')
        feats = ImageFeatures.deserialize(json.loads(message_str))
        self.assertTrue(isinstance(feats, ImageFeatures))
        self.assertFalse(feats.valid_rowcol)

    def test_string_store_load(self):

        data = json.dumps({'a': 1, 'b': 2})

        self.storage.store_string(self.tmp_json_key, data)
        data2 = self.storage.load_string(self.tmp_json_key)
        self.assertEqual(data, data2)

    def test_delete(self):

        data = json.dumps({'a': 1, 'b': 2})
        self.storage.store_string(self.tmp_json_key, data)
        self.storage.delete(self.tmp_json_key)
        self.assertIsNone(self.bucket.get_key(self.tmp_json_key))

    def test_load_legacy_model(self):
        clf = self.storage.load_classifier('legacy.model')
        self.assertTrue(isinstance(clf, CalibratedClassifierCV))

    def test_load_store_model(self):
        clf = CalibratedClassifierCV(SGDClassifier())
        self.storage.store_classifier(self.tmp_model_key, clf)
        self.assertTrue(self.storage.exists(self.tmp_model_key))

        clf2 = self.storage.load_classifier(self.tmp_model_key)
        self.assertTrue(isinstance(clf2, CalibratedClassifierCV))




