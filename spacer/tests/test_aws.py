import unittest
from unittest import mock

from botocore.exceptions import NoCredentialsError

from spacer.aws import aws_check, get_s3_resource, THREAD_LOCAL


class TestGetS3Resource(unittest.TestCase):

    called_str = "Called boto3.Session.resource() in get_s3_resource()"

    def setUp(self):
        # Each test will start out with no S3 resource retrieved yet.
        if hasattr(THREAD_LOCAL, 's3_resource'):
            delattr(THREAD_LOCAL, 's3_resource')

    def tearDown(self):
        # Clear the resource after the test so that other tests aren't
        # broken by bogus values.
        if hasattr(THREAD_LOCAL, 's3_resource'):
            delattr(THREAD_LOCAL, 's3_resource')

    @staticmethod
    def noop(*args, **kwargs):
        pass

    def test_call_once(self):
        with (
            mock.patch('spacer.aws.create_aws_resource', self.noop),
            self.assertLogs(logger='spacer.aws', level='INFO') as cm,
        ):
            get_s3_resource()
        self.assertIn(
            self.called_str,
            cm.output[0])

    def test_reuse(self):
        """Three get_s3_resource() calls, but only one boto3 call."""
        with (
            mock.patch('spacer.aws.create_aws_resource', self.noop),
            self.assertLogs(logger='spacer.aws', level='INFO') as cm,
        ):
            get_s3_resource()
            get_s3_resource()
            get_s3_resource()
        self.assertIn(
            self.called_str,
            cm.output[0])
        self.assertEqual(len(cm.output), 1)


def mock_create_session():

    class Session:

        @staticmethod
        def client(service_name):
            if service_name != 'sts':
                raise ValueError

            class Client:
                @staticmethod
                def get_caller_identity():
                    return dict(Arn='arn:aws:sts::arn-goes-here')

            return Client()

    return Session()


class TestAwsCheck(unittest.TestCase):

    def setUp(self):
        self.printed_strs = []

    def mock_print(self, s):
        """Save the string instead of printing to stdout."""
        self.printed_strs.append(s)

    def test_anonymous(self):
        with (
            mock.patch('spacer.aws.print', self.mock_print),
            mock.patch('spacer.config.AWS_ANONYMOUS', True),
        ):
            aws_check()

        self.assertListEqual(
            self.printed_strs,
            ["AWS_ANONYMOUS has been set to True, so AWS will be accessed"
             " without credentials."],
        )

    @staticmethod
    def mock_getenv_factory(**env_dict):

        def mock_getenv(requested_key, _default=None):
            for key, value in env_dict.items():
                if requested_key == key:
                    return value
            return _default
        return mock_getenv

    def test_variables_1(self):
        with (
            mock.patch('spacer.aws.print', self.mock_print),
            mock.patch(
                'spacer.aws.create_aws_session',
                mock_create_session),
            mock.patch('spacer.config.AWS_PROFILE_NAME', 'my-profile'),
            mock.patch('spacer.config.AWS_ACCESS_KEY_ID', None),
            mock.patch('spacer.config.AWS_SECRET_ACCESS_KEY', None),
            mock.patch('spacer.config.AWS_SESSION_TOKEN', None),
            mock.patch(
                'os.getenv',
                self.mock_getenv_factory(AWS_CONFIG_FILE='~/aws_config')),
        ):
            aws_check()

        self.assertListEqual(
            self.printed_strs,
            ["Variables that have been set:",
             "Spacer config - AWS_PROFILE_NAME"
             " (highest priority if specified): Yes",
             "Environment var - AWS_CONFIG_FILE: Yes",
             "Environment var - AWS_SHARED_CREDENTIALS_FILE: No",
             "Spacer config - AWS_ACCESS_KEY_ID: No",
             "Spacer config - AWS_SECRET_ACCESS_KEY: No",
             "Spacer config - AWS_SESSION_TOKEN: No",
             "Will now call create_aws_session()...",
             "Will now call get_caller_identity()...",
             "Identified as ARN: arn:aws:sts::arn-goes-here"],
        )

    def test_variables_2(self):
        with (
            mock.patch('spacer.aws.print', self.mock_print),
            mock.patch(
                'spacer.aws.create_aws_session',
                mock_create_session),
            mock.patch('spacer.config.AWS_PROFILE_NAME', None),
            mock.patch('spacer.config.AWS_ACCESS_KEY_ID', 'myid'),
            mock.patch('spacer.config.AWS_SECRET_ACCESS_KEY', 'mykey'),
            mock.patch('spacer.config.AWS_SESSION_TOKEN', 'mytoken'),
            mock.patch(
                'os.getenv',
                self.mock_getenv_factory(AWS_SHARED_CREDENTIALS_FILE='~/aws_credentials')),
        ):
            aws_check()

        self.assertListEqual(
            self.printed_strs,
            ["Variables that have been set:",
             "Spacer config - AWS_PROFILE_NAME"
             " (highest priority if specified): No",
             "Environment var - AWS_CONFIG_FILE: No",
             "Environment var - AWS_SHARED_CREDENTIALS_FILE: Yes",
             "Spacer config - AWS_ACCESS_KEY_ID: Yes",
             "Spacer config - AWS_SECRET_ACCESS_KEY: Yes",
             "Spacer config - AWS_SESSION_TOKEN: Yes",
             "Will now call create_aws_session()...",
             "Will now call get_caller_identity()...",
             "Identified as ARN: arn:aws:sts::arn-goes-here"],
        )


if __name__ == '__main__':
    unittest.main()
