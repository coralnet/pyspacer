import unittest
import warnings
import json

from PIL import Image

from io import BytesIO
from spacer import config
import pickle

import numpy as np
import os

from spacer.byteio_storage import S3Storage, FileSystemStorage
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier


from spacer.data_classes import ImageFeatures
from spacer.messages import DataLocation


@unittest.skip
@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestS3Storage(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.tmp_image_key = 'tmp_image.jpg'
        self.tmp_json_key = 'tmp_data.json'
        self.tmp_model_key = 'tmp_model.pkl'

        conn = config.get_s3_conn()
        self.bucket = conn.get_bucket('spacer-test')

    def tearDown(self):
        self.bucket.delete_key(self.tmp_image_key)
        self.bucket.delete_key(self.tmp_json_key)
        self.bucket.delete_key(self.tmp_model_key)

    def test_image_features(self):

        feats = ImageFeatures.example()
        loc = DataLocation(storage_type='s3',
                           bucket_name='spacer-test',
                           key=self.tmp_json_key)
        storage = S3Storage()

        storage.store(loc, BytesIO(
            json.dumps(feats.serialize()).encode('utf-8')))

        feats2 = ImageFeatures.deserialize(json.loads(
            storage.load(loc).getvalue().decode('utf-8')))

        self.assertEqual(feats, feats2)

    def test_image(self):

        img = Image.new('RGB', (100, 100))
        loc = DataLocation(storage_type='s3',
                           bucket_name='spacer-test',
                           key=self.tmp_image_key)

        storage = S3Storage()

        with BytesIO() as stream:
            img.save(stream, 'JPEG')
            stream.seek(0)
            storage.store(loc, stream)

        img2 = Image.open(storage.load(loc))

        self.assertTrue(np.array_equal(np.array(img), np.array(img2)))

    def test_classifier(self):
        clf = CalibratedClassifierCV(SGDClassifier())
        loc = DataLocation(storage_type='s3',
                           bucket_name='spacer-test',
                           key=self.tmp_model_key)

        storage = S3Storage()
        storage.store(loc, BytesIO(pickle.dumps(clf, protocol=2)))

        pickle.loads(storage.load(loc).getbuffer(), fix_imports=True,
                     encoding='latin1')

@unittest.skip
class TestFileSystemStorage(unittest.TestCase):

    def setUp(self):
        self.tmp_image_file_name = 'tmp_image.jpg'
        self.tmp_json_file_name = 'tmp_data.json'
        self.tmp_model_file_name = 'tmp_model.pkl'

    def tearDown(self):

        for tmp_file in [self.tmp_json_file_name,
                         self.tmp_image_file_name,
                         self.tmp_model_file_name]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    def test_image_features(self):

        feats = ImageFeatures.example()
        loc = DataLocation(storage_type='filesystem',
                           bucket_name='spacer-test',
                           key=self.tmp_json_file_name)
        storage = FileSystemStorage()

        storage.store(loc, BytesIO(
            json.dumps(feats.serialize()).encode('utf-8')))

        feats2 = ImageFeatures.deserialize(json.loads(
            storage.load(loc).getvalue().decode('utf-8')))

        self.assertEqual(feats, feats2)

    def test_image(self):

        img = Image.new('RGB', (100, 100))
        loc = DataLocation(storage_type='s3',
                           bucket_name='spacer-test',
                           key=self.tmp_image_key)

        storage = S3Storage()

        with BytesIO() as stream:
            img.save(stream, 'JPEG')
            stream.seek(0)
            storage.store(loc, stream)

        img2 = Image.open(storage.load(loc))

        self.assertTrue(np.array_equal(np.array(img), np.array(img2)))

    def test_classifier(self):
        clf = CalibratedClassifierCV(SGDClassifier())
        loc = DataLocation(storage_type='s3',
                           bucket_name='spacer-test',
                           key=self.tmp_model_key)

        storage = S3Storage()
        storage.store(loc, BytesIO(pickle.dumps(clf, protocol=2)))

        pickle.loads(storage.load(loc).getbuffer(), fix_imports=True,
                     encoding='latin1')