import unittest

from spacer import config


require_caffe = unittest.skipUnless(
    config.HAS_CAFFE, "Requires Caffe to be installed")

require_cn_test_extractors = unittest.skipUnless(
    config.CN_TEST_EXTRACTORS_BUCKET,
    "Requires access to the test feature-extractors on CoralNet's S3")

require_cn_fixtures = unittest.skipUnless(
    config.CN_FIXTURES_BUCKET,
    "Requires access to the test fixtures on CoralNet's S3")

require_s3 = unittest.skipUnless(
    config.TEST_BUCKET,
    "Requires write access to an S3 bucket for use in tests")
