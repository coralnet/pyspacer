import abc
import json
import os
import socket
import time
import unittest
import urllib.request
from http.client import HTTPMessage, IncompleteRead
from io import BytesIO
from unittest import mock
from urllib.error import HTTPError

import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV

from spacer import config
from spacer.data_classes import ImageFeatures
from spacer.exceptions import URLDownloadError
from spacer.messages import DataLocation
from spacer.storage import \
    storage_factory, \
    load_image, \
    load_classifier, \
    store_image, \
    store_classifier, \
    clear_memory_storage
from spacer.tests.utils import cn_beta_fixture_location
from spacer.train_utils import make_random_data, train
from .decorators import require_test_fixtures
from .utils import temp_filesystem_data_location


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


class BaseStorageTest(unittest.TestCase, abc.ABC):

    def do_test_delete(self):
        data = json.dumps({'a': 1, 'b': 2})
        stream = BytesIO(json.dumps(data).encode('utf-8'))
        self.storage.store(self.tmp_json_loc.key, stream)
        self.assertTrue(self.storage.exists(self.tmp_json_loc.key))
        self.storage.delete(self.tmp_json_loc.key)
        self.assertFalse(self.storage.exists(self.tmp_json_loc.key))

    def do_test_load_store_model(self):
        features_loc_template = DataLocation(storage_type='memory', key='')
        train_labels = make_random_data(
            im_count=10,
            class_list=[1, 2],
            points_per_image=5,
            feature_dim=5,
            feature_loc=features_loc_template,
        )
        ref_labels = make_random_data(
            im_count=2,
            class_list=[1, 2],
            points_per_image=5,
            feature_dim=5,
            feature_loc=features_loc_template,
        )
        clf, _ = train(
            train_labels, ref_labels, features_loc_template, 1, 'LR')
        store_classifier(self.tmp_model_loc, clf)
        self.assertTrue(self.storage.exists(self.tmp_model_loc.key))

        clf2 = load_classifier(self.tmp_model_loc)
        self.assertTrue(isinstance(clf2, CalibratedClassifierCV))


def raise_404(url, *args, **kwargs):
    raise HTTPError(url, 404, "Not found", HTTPMessage(), None)


def raise_timeout(url, *args, **kwargs):
    # When only supporting Python 3.10+, change socket.timeout to TimeoutError
    raise socket.timeout("timed out")


class TestURLStorage(unittest.TestCase):

    INVALID_URL = 'not_even_a_url'
    UNREACHABLE_DOMAIN = 'https://not-a-real-domain/'

    @classmethod
    def setUpClass(cls):
        cls.storage = storage_factory('url')

    @staticmethod
    def s3_url(filepath):
        return (
            'https://'
            f'{config.TEST_BUCKET}.s3-{config.AWS_REGION}.amazonaws.com/'
            f'{filepath}'
        )

    @require_test_fixtures
    def test_load_image(self):
        loc = DataLocation(
            storage_type='url',
            key=self.s3_url('08bfc10v7t.png'),
        )
        img = load_image(loc)
        self.assertTrue(isinstance(img, Image.Image))

    @require_test_fixtures
    def test_load_classifier(self):
        loc = DataLocation(
            storage_type='url',
            key=self.s3_url('legacy_compat/coralnet_beta/example.model'),
        )
        clf = load_classifier(loc)
        self.assertTrue(isinstance(clf, CalibratedClassifierCV))

    @require_test_fixtures
    def test_load_string(self):
        loc = DataLocation(
            storage_type='url',
            key=self.s3_url('08bfc10v7t.png.featurevector'),
        )
        feats = ImageFeatures.load(loc)
        self.assertTrue(isinstance(feats, ImageFeatures))

    @require_test_fixtures
    def test_exists_true(self):
        self.assertTrue(self.storage.exists(self.s3_url('08bfc10v7t.png')))

    def test_exists_false(self):
        self.assertFalse(self.storage.exists(self.INVALID_URL))
        self.assertFalse(self.storage.exists(self.UNREACHABLE_DOMAIN))

        with mock.patch('urllib.request.urlopen', raise_404):
            self.assertFalse(self.storage.exists('a_url'))

        with mock.patch('urllib.request.urlopen', raise_timeout):
            self.assertFalse(self.storage.exists('a_url'))

    def test_unsupported_methods(self):
        self.assertRaises(TypeError,
                          self.storage.store,
                          'dummy',
                          Image.new('RGB', (200, 200)))

        self.assertRaises(TypeError,
                          self.storage.delete,
                          'dummy')

    def test_invalid_url(self):
        with self.assertRaises(URLDownloadError) as context:
            self.storage.load(self.INVALID_URL)
        self.assertEqual(
            f"Failed to download from the URL '{self.INVALID_URL}'."
            f" / Details - ValueError: unknown url type: '{self.INVALID_URL}'",
            str(context.exception),
            "Should raise the appropriate error",
        )

    def test_unreachable_domain(self):
        with self.assertRaises(URLDownloadError) as cm:
            self.storage.load(self.UNREACHABLE_DOMAIN)
        # "getaddrinfo" on Windows or "No address" on Linux
        self.assertIn("addr", str(cm.exception))

    def test_404(self):
        with mock.patch('urllib.request.urlopen', raise_404):
            with self.assertRaises(URLDownloadError) as cm:
                self.storage.load('a_url')
        self.assertIn("404", str(cm.exception))

    def test_timeout(self):
        with mock.patch('urllib.request.urlopen', raise_timeout):
            with self.assertRaises(URLDownloadError) as cm:
                self.storage.load('a_url')
        self.assertIn("timed out", str(cm.exception))

    def test_incomplete_read(self):
        class FakeResponse:
            def read(self):
                raise IncompleteRead(b'')

        def return_fake_response(*args, **kwargs):
            return FakeResponse()

        with mock.patch.object(
            urllib.request, 'urlopen', return_fake_response
        ):
            with self.assertRaises(URLDownloadError) as cm:
                self.storage.load('url')
        self.assertIn("full response", str(cm.exception))


@require_test_fixtures
class TestS3Storage(BaseStorageTest):

    def setUp(self):
        config.filter_warnings()

        self.tmp_image_loc = DataLocation(
            storage_type='s3',
            key='tmp_image.jpg',
            bucket_name=config.TEST_BUCKET
        )
        self.tmp_json_loc = DataLocation(
            storage_type='s3',
            key='tmp_data.json',
            bucket_name=config.TEST_BUCKET
        )
        self.tmp_model_loc = DataLocation(
            storage_type='s3',
            key='tmp_model.pkl',
            bucket_name=config.TEST_BUCKET
        )
        self.storage = storage_factory('s3', config.TEST_BUCKET)

    def tearDown(self):
        self.storage.delete(self.tmp_image_loc.key)
        self.storage.delete(self.tmp_json_loc.key)
        self.storage.delete(self.tmp_model_loc.key)

    def test_load_store_image(self):
        img = Image.new('RGB', (100, 200))
        store_image(self.tmp_image_loc, img)
        img2 = load_image(self.tmp_image_loc)

        self.assertTrue(np.allclose(np.array(img), np.array(img2), atol=1e-5))
        self.assertTrue(isinstance(img2, Image.Image))

    def test_load_legacy_features(self):
        feats = ImageFeatures.load(
            cn_beta_fixture_location('example.jpg.feats'))
        self.assertTrue(isinstance(feats, ImageFeatures))
        self.assertFalse(feats.valid_rowcol)

    def test_delete(self):
        self.do_test_delete()

    def test_load_legacy_model(self):
        clf = load_classifier(cn_beta_fixture_location('example.model'))
        self.assertTrue(isinstance(clf, CalibratedClassifierCV))

    def test_load_store_model(self):
        self.do_test_load_store_model()


class TestLocalStorage(BaseStorageTest):

    def setUp(self):
        self.storage = storage_factory('filesystem', '')

    def test_load_store_image(self):
        img = Image.new('RGB', (100, 200))

        with temp_filesystem_data_location() as tmp_image_loc:
            store_image(tmp_image_loc, img)
            img2 = load_image(tmp_image_loc)

        self.assertTrue(np.array_equal(np.array(img), np.array(img2)))
        self.assertTrue(isinstance(img2, Image.Image))

    def test_load_legacy_features(self):
        loc = DataLocation(
            storage_type='filesystem',
            key=os.path.join(config.LOCAL_FIXTURE_DIR, 'cnbeta.jpg.feats')
        )
        feats = ImageFeatures.load(loc)
        self.assertTrue(isinstance(feats, ImageFeatures))
        self.assertFalse(feats.valid_rowcol)

    def test_string_store_load(self):

        data = json.dumps({'a': 1, 'b': 2})
        stream = BytesIO(json.dumps(data).encode('utf-8'))

        with temp_filesystem_data_location() as tmp_json_loc:
            self.storage.store(tmp_json_loc.key, stream)
            data2 = json.loads(self.storage.load(
                tmp_json_loc.key).getvalue().decode('utf-8'))

        self.assertEqual(data, data2)

    def test_delete(self):
        with temp_filesystem_data_location() as temp_loc:
            self.tmp_json_loc = temp_loc
            self.do_test_delete()

    def test_load_legacy_model(self):
        loc = DataLocation(
            storage_type='filesystem',
            key=os.path.join(config.LOCAL_FIXTURE_DIR, 'cnbeta.model')
        )
        clf = load_classifier(loc)
        self.assertTrue(isinstance(clf, CalibratedClassifierCV))

    def test_load_store_model(self):
        with temp_filesystem_data_location() as temp_loc:
            self.tmp_model_loc = temp_loc
            self.do_test_load_store_model()


class TestMemoryStorage(BaseStorageTest):

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
        self.do_test_delete()

    def test_load_store_model(self):
        self.do_test_load_store_model()


class TestFactory(unittest.TestCase):

    def test_bad_storage_type(self):
        self.assertRaises(AssertionError,
                          storage_factory,
                          'not_a_valid_storage')


class TestLRUCache(unittest.TestCase):

    def test_load_classifier(self):
        loc = DataLocation(
            storage_type='filesystem',
            key=os.path.join(config.LOCAL_FIXTURE_DIR, 'cnbeta.model')
        )

        load_classifier.cache_clear()
        t0 = time.time()
        load_classifier(loc)
        t1 = time.time() - t0

        t0 = time.time()
        load_classifier(loc)
        t2 = time.time() - t0
        self.assertLess(t2, t1)


if __name__ == '__main__':
    unittest.main()
