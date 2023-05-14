import abc
import json
import os
import time
import unittest
from io import BytesIO

import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV

from spacer import config
from spacer.data_classes import ImageFeatures
from spacer.exceptions import SpacerInputError
from spacer.messages import DataLocation
from spacer.storage import \
    storage_factory, \
    download_model, \
    load_image, \
    load_classifier, \
    store_image, \
    store_classifier, \
    clear_memory_storage
from spacer.tests.utils import cn_beta_fixture_location
from spacer.train_utils import make_random_data, train
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
            im_count=20,
            class_list=[1, 2],
            points_per_image=5,
            feature_dim=5,
            feature_loc=features_loc_template,
        )
        clf, _ = train(train_labels, features_loc_template, 1, 'LR')
        store_classifier(self.tmp_model_loc, clf)
        self.assertTrue(self.storage.exists(self.tmp_model_loc.key))

        clf2 = load_classifier(self.tmp_model_loc)
        self.assertTrue(isinstance(clf2, CalibratedClassifierCV))


class TestURLStorage(unittest.TestCase):

    INVALID_URL = 'not_even_a_url'
    UNREACHABLE_DOMAIN = 'https://not-a-real-domain/'
    UNREACHABLE_URL = 'https://coralnet.ucsd.edu/not-a-real-page/'

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
            key='https://spacer-test.s3-us-west-2.amazonaws.com/'
                'legacy_compat/coralnet_beta/example.model'
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
        self.assertFalse(self.storage.exists(self.INVALID_URL))
        self.assertFalse(self.storage.exists(self.UNREACHABLE_DOMAIN))
        self.assertFalse(self.storage.exists(self.UNREACHABLE_URL))

    def test_unsupported_methods(self):
        self.assertRaises(TypeError,
                          self.storage.store,
                          'dummy',
                          Image.new('RGB', (200, 200)))

        self.assertRaises(TypeError,
                          self.storage.delete,
                          'dummy')

    def test_invalid_url(self):
        with self.assertRaises(SpacerInputError) as context:
            self.storage.load(self.INVALID_URL)
        self.assertEqual(
            f"unknown url type: '{self.INVALID_URL}'",
            context.exception.args[0],
            "Should raise the appropriate error",
        )

    def test_unreachable_domain(self):
        with self.assertRaises(SpacerInputError):
            self.storage.load(self.UNREACHABLE_DOMAIN)

    def test_unreachable_url(self):
        with self.assertRaises(SpacerInputError):
            self.storage.load(self.UNREACHABLE_URL)


@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
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


class TestDownloadModel(unittest.TestCase):

    @unittest.skipUnless(config.HAS_S3_MODEL_ACCESS, 'No access to models')
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
