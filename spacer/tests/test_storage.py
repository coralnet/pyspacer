import json
import os
import time
import unittest
from io import BytesIO

import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier

from spacer import config
from spacer.data_classes import ImageFeatures
from spacer.messages import DataLocation
from spacer.storage import \
    storage_factory, \
    download_model, \
    load_image, \
    load_classifier, \
    store_image, \
    store_classifier, \
    clear_memory_storage


class TestGlobalMemoryStorage(unittest.TestCase):

    def test_simple(self):

        clear_memory_storage()

        # Check that memory storage is global.
        storage = storage_factory('memory')
        self.assertFalse(storage.exists('feats'))

        ImageFeatures.example().store(DataLocation(storage_type='memory',
                                                   key='feats'))
        self.assertTrue(storage.exists('feats'))

        # Deleting the local pointer doesn't erase the memory.
        del storage
        storage = storage_factory('memory')
        self.assertTrue(storage.exists('feats'))

        # Calling the cleanup will, however
        clear_memory_storage()
        storage = storage_factory('memory')
        self.assertFalse(storage.exists('feats'))


class TestURLStorage(unittest.TestCase):

    def setUp(self):
        self.storage = storage_factory('url')

    def test_load_image(self):
        loc = DataLocation(
            storage_type='url',
            key='https://spacer-test.s3-us-west-2.amazonaws.com/08bfc10v7t.png'
        )
        img = load_image(loc)
        self.assertTrue(isinstance(img, Image.Image))

    def test_load_classifier(self):
        loc = DataLocation(
            storage_type='url',
            key='https://spacer-test.s3-us-west-2.amazonaws.com/legacy.model'
        )
        clf = load_classifier(loc)
        self.assertTrue(isinstance(clf, CalibratedClassifierCV))

    def test_load_string(self):
        loc = DataLocation(
            storage_type='url',
            key='https://spacer-test.s3-us-west-2.amazonaws.com/'
            '08bfc10v7t.png.featurevector'
        )
        feats = ImageFeatures.load(loc)
        self.assertTrue(isinstance(feats, ImageFeatures))

    def test_exists(self):
        self.assertTrue(self.storage.exists(
            'https://spacer-test.s3-us-west-2.amazonaws.com/08bfc10v7t.png'))
        self.assertFalse(self.storage.exists(
            'not_even_a_url'
        ))
        self.assertFalse(self.storage.exists(
            'https://not-a-real-domain/image.png'
        ))

    def test_unsupported_methods(self):
        self.assertRaises(TypeError,
                          self.storage.store,
                          'dummy',
                          Image.new('RGB', (200, 200)))

        self.assertRaises(TypeError,
                          self.storage.delete,
                          'dummy')


@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestS3Storage(unittest.TestCase):

    def setUp(self):
        config.filter_warnings()

        self.tmp_image_loc = DataLocation(
            storage_type='s3',
            key='tmp_image.jpg',
            bucket_name='spacer-test'
        )
        self.tmp_json_loc = DataLocation(
            storage_type='s3',
            key='tmp_data.json',
            bucket_name='spacer-test'
        )
        self.tmp_model_loc = DataLocation(
            storage_type='s3',
            key='tmp_model.pkl',
            bucket_name='spacer-test'
        )
        self.storage = storage_factory('s3', 'spacer-test')
        conn = config.get_s3_conn()
        self.bucket = conn.get_bucket('spacer-test')

    def tearDown(self):
        self.bucket.delete_key(self.tmp_image_loc.key)
        self.bucket.delete_key(self.tmp_json_loc.key)
        self.bucket.delete_key(self.tmp_model_loc.key)

    def test_load_store_image(self):

        img = Image.new('RGB', (100, 200))
        store_image(self.tmp_image_loc, img)
        img2 = load_image(self.tmp_image_loc)

        self.assertTrue(np.allclose(np.array(img), np.array(img2), atol=1e-5))
        self.assertTrue(isinstance(img2, Image.Image))

    def test_load_legacy_features(self):
        feats = ImageFeatures.load(DataLocation(
            storage_type='s3',
            key='legacy.jpg.feats',
            bucket_name='spacer-test'
        ))
        self.assertTrue(isinstance(feats, ImageFeatures))
        self.assertFalse(feats.valid_rowcol)

    def test_delete(self):
        store_image(self.tmp_image_loc, Image.new('RGB', (100, 100)))
        self.storage.delete(self.tmp_json_loc.key)
        self.assertIsNone(self.bucket.get_key(self.tmp_json_loc.key))

    def test_load_legacy_model(self):
        clf = load_classifier(DataLocation(
            storage_type='s3',
            key='legacy.model',
            bucket_name='spacer-test'
        ))
        self.assertTrue(isinstance(clf, CalibratedClassifierCV))

    def test_load_store_model(self):
        clf = CalibratedClassifierCV(SGDClassifier())
        store_classifier(self.tmp_model_loc, clf)
        self.assertTrue(self.storage.exists(self.tmp_model_loc.key))

        clf2 = load_classifier(self.tmp_model_loc)
        self.assertTrue(isinstance(clf2, CalibratedClassifierCV))


class TestLocalStorage(unittest.TestCase):

    def setUp(self):
        self.tmp_image_loc = DataLocation(
            storage_type='filesystem',
            key='tmp_image.jpg',
            bucket_name=''
        )
        self.tmp_json_loc = DataLocation(
            storage_type='filesystem',
            key='tmp_data.json',
            bucket_name=''
        )
        self.tmp_model_loc = DataLocation(
            storage_type='filesystem',
            key='tmp_model.pkl',
            bucket_name=''
        )
        self.storage = storage_factory('filesystem', '')

    def tearDown(self):

        for tmp_loc in [self.tmp_json_loc,
                        self.tmp_image_loc,
                        self.tmp_model_loc]:
            if os.path.exists(tmp_loc.key):
                os.remove(tmp_loc.key)

    def test_load_store_image(self):

        img = Image.new('RGB', (100, 200))
        store_image(self.tmp_image_loc, img)
        img2 = load_image(self.tmp_image_loc)
        self.assertTrue(np.array_equal(np.array(img), np.array(img2)))
        self.assertTrue(isinstance(img2, Image.Image))

    def test_load_legacy_features(self):
        loc = DataLocation(
            storage_type='filesystem',
            key=os.path.join(config.LOCAL_FIXTURE_DIR, 'legacy.jpg.feats')
        )
        feats = ImageFeatures.load(loc)
        self.assertTrue(isinstance(feats, ImageFeatures))
        self.assertFalse(feats.valid_rowcol)

    def test_string_store_load(self):

        data = json.dumps({'a': 1, 'b': 2})
        stream = BytesIO(json.dumps(data).encode('utf-8'))
        self.storage.store(self.tmp_json_loc.key, stream)

        data2 = json.loads(self.storage.load(
            self.tmp_json_loc.key).getvalue().decode('utf-8'))
        self.assertEqual(data, data2)

    def test_delete(self):

        data = json.dumps({'a': 1, 'b': 2})
        stream = BytesIO(json.dumps(data).encode('utf-8'))
        self.storage.store(self.tmp_json_loc.key, stream)
        self.storage.delete(self.tmp_json_loc.key)
        self.assertFalse(os.path.exists(self.tmp_json_loc.key))

    def test_load_legacy_model(self):
        loc = DataLocation(
            storage_type='filesystem',
            key=os.path.join(config.LOCAL_FIXTURE_DIR, 'legacy.model')
        )
        clf = load_classifier(loc)
        self.assertTrue(isinstance(clf, CalibratedClassifierCV))

    def test_load_store_model(self):
        clf = CalibratedClassifierCV(SGDClassifier())
        store_classifier(self.tmp_model_loc, clf)
        self.assertTrue(self.storage.exists(self.tmp_model_loc.key))

        clf2 = load_classifier(self.tmp_model_loc)
        self.assertTrue(isinstance(clf2, CalibratedClassifierCV))


class TestMemoryStorage(unittest.TestCase):

    def setUp(self):
        self.tmp_image_loc = DataLocation(
            storage_type='memory',
            key='tmp_image.jpg',
            bucket_name=''
        )
        self.tmp_json_loc = DataLocation(
            storage_type='memory',
            key='tmp_data.json',
            bucket_name=''
        )
        self.tmp_model_loc = DataLocation(
            storage_type='memory',
            key='tmp_model.pkl',
            bucket_name=''
        )
        self.storage = storage_factory('memory', '')

    def test_load_store_image(self):
        img = Image.new('RGB', (100, 200))
        store_image(self.tmp_image_loc, img)
        img2 = load_image(self.tmp_image_loc)
        self.assertTrue(np.array_equal(np.array(img), np.array(img2)))
        self.assertTrue(isinstance(img2, Image.Image))

    def test_string_store_load(self):
        data = ImageFeatures.example()
        data.store(self.tmp_json_loc)
        data2 = ImageFeatures.load(self.tmp_json_loc)
        self.assertEqual(data, data2)

    def test_delete(self):
        data = json.dumps({'a': 1, 'b': 2})
        stream = BytesIO(json.dumps(data).encode('utf-8'))
        self.storage.store(self.tmp_json_loc.key, stream)
        self.assertTrue(self.storage.exists(self.tmp_json_loc.key))
        self.storage.delete(self.tmp_json_loc.key)
        self.assertFalse(self.storage.exists(self.tmp_json_loc.key))

    def test_load_store_model(self):
        clf = CalibratedClassifierCV(SGDClassifier())
        store_classifier(self.tmp_model_loc, clf)
        self.assertTrue(self.storage.exists(self.tmp_model_loc.key))

        clf2 = load_classifier(self.tmp_model_loc)
        self.assertTrue(isinstance(clf2, CalibratedClassifierCV))


class TestFactory(unittest.TestCase):

    def test_bad_storage_type(self):

        self.assertRaises(AssertionError,
                          storage_factory,
                          'not_a_valid_storage')


class TestDownloadModel(unittest.TestCase):

    @unittest.skipUnless(config.HAS_S3_MODEL_ACCESS, 'No access to models')
    @unittest.skipUnless(config.HAS_LOCAL_MODEL_PATH,
                         'Local model path not set or is invalid.')
    def test_ok(self):
        keyname = 'vgg16_coralnet_ver1.deploy.prototxt'
        destination = os.path.join(config.LOCAL_MODEL_PATH, keyname)
        storage = storage_factory('filesystem')
        if storage.exists(destination):
            storage.delete(destination)

        destination_, was_cached = download_model(keyname)
        self.assertFalse(was_cached)
        self.assertTrue(storage.exists(destination))
        self.assertEqual(destination_, destination)

        destination_, was_cached = download_model(keyname)
        self.assertTrue(was_cached)


class TestLRUCache(unittest.TestCase):

    def test_load_classifier(self):
        loc = DataLocation(
            storage_type='filesystem',
            key=os.path.join(config.LOCAL_FIXTURE_DIR, 'legacy.model')
        )

        load_classifier.cache_clear()
        t0 = time.time()
        load_classifier(loc)
        t1 = time.time()-t0

        t0 = time.time()
        load_classifier(loc)
        t2 = time.time() - t0
        self.assertLess(t2, t1)
