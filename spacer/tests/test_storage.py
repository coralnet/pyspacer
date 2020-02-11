import json
import os
import unittest
import warnings
from io import BytesIO

import boto
from PIL import Image

from spacer import config
from spacer.storage import storage_factory


class TestLocalStorage(unittest.TestCase):

    def setUp(self):
        self.tmp_image_file_name = 'tmp_image.jpg'
        self.tmp_json_file_name = 'tmp_data.json'
        self.storage = storage_factory('local', '')

    def tearDown(self):
        if os.path.exists(self.tmp_image_file_name):
            os.remove(self.tmp_image_file_name)

        if os.path.exists(self.tmp_json_file_name):
            os.remove(self.tmp_json_file_name)

    def test_load_image(self):

        img = Image.new('RGB', (100, 200))
        img.save(self.tmp_image_file_name)

        img2 = self.storage.load_image(self.tmp_image_file_name)

        self.assertEqual(img.size[0], img2.size[0])
        self.assertEqual(img.size[1], img2.size[1])

    def test_string_store_load(self):

        data = json.dumps({'a': 1, 'b': 2})

        self.storage.store_string(data, self.tmp_json_file_name)
        data2 = self.storage.load_string(self.tmp_json_file_name)
        self.assertEqual(data, data2)

    def test_delete(self):

        data = json.dumps({'a': 1, 'b': 2})
        self.storage.store_string(data, self.tmp_json_file_name)
        self.storage.delete(self.tmp_json_file_name)
        self.assertFalse(os.path.exists(self.tmp_json_file_name))


@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestS3Storage(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.tmp_image_key = 'tmp_image.jpg'
        self.tmp_json_key = 'tmp_data.json'
        self.storage = storage_factory('s3', 'spacer-test')

        conn = boto.connect_s3()
        self.bucket = conn.get_bucket('spacer-test')

    def tearDown(self):
        self.bucket.delete_key(self.tmp_image_key)
        self.bucket.delete_key(self.tmp_json_key)

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

    def test_string_store_load(self):

        data = json.dumps({'a': 1, 'b': 2})

        self.storage.store_string(data, self.tmp_json_key)
        data2 = self.storage.load_string(self.tmp_json_key)
        self.assertEqual(data, data2)

    def test_delete(self):

        data = json.dumps({'a': 1, 'b': 2})
        self.storage.store_string(data, self.tmp_json_key)
        self.storage.delete(self.tmp_json_key)
        self.assertIsNone(self.bucket.get_key(self.tmp_json_key))




