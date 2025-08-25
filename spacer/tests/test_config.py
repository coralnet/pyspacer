import unittest
from unittest import mock

from spacer.config import get_config_value
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


if __name__ == '__main__':
    unittest.main()
