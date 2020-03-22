import unittest
import warnings
import json

from io import BytesIO
from spacer import config

from spacer.byteio_storage import S3Storage

from spacer.data_classes import ImageFeatures, DataLocation


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
        storage.store(loc, BytesIO(json.dumps(feats.serialize())))