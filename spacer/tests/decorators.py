import unittest

from spacer import config


require_caffe = unittest.skipUnless(
    config.HAS_CAFFE, "Requires Caffe to be installed")

require_test_extractors = unittest.skipUnless(
    config.TEST_EXTRACTORS_BUCKET,
    "Requires access to the test feature-extractors on S3")

require_test_fixtures = unittest.skipUnless(
    config.TEST_BUCKET,
    "Requires access to the test fixtures on S3")
