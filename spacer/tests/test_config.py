import unittest
from unittest import mock

from spacer.config import get_config_value, get_s3_resource, THREAD_LOCAL
from spacer.exceptions import ConfigError


def mock_getenv_factory(key, value):

    def mock_getenv(key_, _default=None):
        if key_ == 'SPACER_' + key:
            return value
    return mock_getenv


class TestGetConfigValue(unittest.TestCase):

    def test_str(self):
        with mock.patch('os.getenv', mock_getenv_factory('KEY', 'test value')):
            self.assertEqual(get_config_value('KEY'), 'test value')

    def test_int_from_getenv(self):
        """
        If a config var comes from getenv or secrets, the code has to ensure
        int config vars are cast from str to int.
        """
        with mock.patch('os.getenv', mock_getenv_factory('KEY', '100')):
            self.assertEqual(get_config_value('KEY', value_type=int), 100)

    def test_not_defined(self):
        """
        os.getenv() returns None if the env var isn't defined.
        """
        with mock.patch('os.getenv', mock_getenv_factory('KEY', None)):
            with self.assertRaises(ConfigError):
                get_config_value('KEY')

    def test_empty_str(self):
        """
        We treat an empty str the same as not defined.
        """
        with mock.patch('os.getenv', mock_getenv_factory('KEY', '')):
            with self.assertRaises(ConfigError):
                get_config_value('KEY')

    def test_not_defined_but_has_default(self):
        with mock.patch('os.getenv', mock_getenv_factory('KEY', None)):
            self.assertEqual(get_config_value('KEY', default='def.'), 'def.')


def mock_boto3_client_factory(status_code):

    def mock_boto3_client(service_name):
        if service_name != 'sts':
            raise ValueError

        class Client:
            @staticmethod
            def get_caller_identity():
                return dict(ResponseMetadata=dict(HTTPStatusCode=status_code))
        return Client()
    return mock_boto3_client


class TestGetS3Resource(unittest.TestCase):

    def setUp(self):
        # Each test will start out with no S3 resource retrieved yet.
        if hasattr(THREAD_LOCAL, 's3_resource'):
            delattr(THREAD_LOCAL, 's3_resource')

    def test_sts(self):
        with mock.patch('boto3.client', mock_boto3_client_factory(200)):
            with self.assertLogs(logger='spacer.config', level='INFO') as cm:
                get_s3_resource()
        self.assertIn(
            "Called boto3.resource() in get_s3_resource(),"
            " with STS credentials",
            cm.output[0])

    def test_sts_failure_response_code(self):
        """Should fall back to spacer config / auto-detect."""
        with mock.patch('boto3.client', mock_boto3_client_factory(400)):
            with self.assertLogs(logger='spacer.config', level='INFO') as cm:
                get_s3_resource()
        self.assertIn(
            "Called boto3.resource() in get_s3_resource(),"
            " with spacer config or auto-detected credentials",
            cm.output[0])

    def test_spacer_config(self):
        with self.assertLogs(logger='spacer.config', level='INFO') as cm:
            get_s3_resource()
        self.assertIn(
            "Called boto3.resource() in get_s3_resource(),"
            " with spacer config or auto-detected credentials",
            cm.output[0])

    def test_reuse(self):
        """Three get_s3_resource() calls, but only one boto3.resource() call."""
        with self.assertLogs(logger='spacer.config', level='INFO') as cm:
            get_s3_resource()
            get_s3_resource()
            get_s3_resource()
        self.assertIn(
            "Called boto3.resource() in get_s3_resource()",
            cm.output[0])
        self.assertEqual(len(cm.output), 1)


if __name__ == '__main__':
    unittest.main()
