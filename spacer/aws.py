import os
import threading
from logging import getLogger

import boto3
import botocore
from botocore.config import Config as BotoConfig
import botocore.exceptions

from spacer import config

logger = getLogger(__name__)


# Save S3 resources for reuse, but only have one per thread,
# because they're not thread-safe:
# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/resources.html
THREAD_LOCAL = threading.local()


def create_aws_session():
    """
    Create an AWS session to encapsulate credentials. This can then be used
    to create a boto client or resource.

    This logic is largely from django-storages' backends/s3.py.
    """
    if config.AWS_PROFILE_NAME:
        # AWS profile name specified in spacer config (env vars, etc).
        # We give this precedence over other
        # credential specifiers such as environment variables.
        return boto3.Session(profile_name=config.AWS_PROFILE_NAME)

    # Use credentials from spacer config (env vars, etc).
    # If credentials are None, boto will instead look in other places such
    # as env var AWS_SHARED_CREDENTIALS_FILE (default ~/.aws/credentials).
    # If no static credentials are defined, then the instance metadata
    # service will be accessed for temporary credentials instead.
    # https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-authentication.html#cli-chap-authentication-precedence
    return boto3.Session(
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        aws_session_token=config.AWS_SESSION_TOKEN,
    )


def create_aws_resource(service_name, region=None):

    if config.AWS_ANONYMOUS:
        # Access public S3 files without credentials of any kind.
        boto_config = BotoConfig(signature_version=botocore.UNSIGNED)
    else:
        boto_config = None

    session = create_aws_session()
    return session.resource(
        service_name,
        region_name=region or config.AWS_REGION,
        config=boto_config,
    )


def get_s3_resource():
    """
    Returns a boto s3 Resource.
    Each thread only gets this from the boto API once, saving it to
    THREAD_LOCAL.s3_resource and reusing it thereafter.

    The logging statements aim to confirm:
    - That each thread only gets one resource (log with %(thread)d
      in the logging format to confirm this)
    - How long a single resource retrieval can be reused before expiring,
      if it ever expires (we might not handle this case yet, but logging
      with timestamp will help confirm the time till expiry)
    """
    try:
        # Reuse this thread's previously established resource, if any.
        return THREAD_LOCAL.s3_resource
    except AttributeError:
        # No resource for this thread yet.
        pass

    THREAD_LOCAL.s3_resource = create_aws_resource('s3')
    logger.info(
        "Called boto3.Session.resource() in get_s3_resource()")
    return THREAD_LOCAL.s3_resource


def aws_check():
    """
    Bit of debug info regarding the AWS credentials that pyspacer is expected
    to use.

    Warning: the print output of this function may be sensitive, including
    locations of static credentials files, or IDs of AWS accounts.
    """
    if config.AWS_ANONYMOUS:
        print(
            "AWS_ANONYMOUS has been set to True, so AWS will be accessed"
            " without credentials.")
        return

    # Print status of various AWS variables.
    # We won't be too specific about how they might be used, because
    # there are many ways to use AWS configuration and credentials.
    # However, we'll list profile name first and static credentials last,
    # to hint at what's considered better security practice.

    print("Variables that have been set:")

    # Observations on priority:
    # - If profile name is specified, it'll totally rely on the AWS config
    #   file. That means: if auth stuff for this profile is present in
    #   that config file, then it's used; if the config file doesn't have
    #   this profile, then it errors; if the config file isn't found, then
    #   it errors.
    # - If profile name is not specified, but explicit key/secret/token
    #   are specified: it'll use that explicit key/secret/token.
    # - If neither of those are specified: it'll look for an AWS
    #   credentials file and use its default-profile entry if it exists,
    #   else it'll look for an AWS config file and use its default-profile
    #   entry if it exists.
    # - Else it'll try other methods (like seeing if it's on EC2 with an
    #   instance role) before giving up with NoCredentialsError.
    #
    # Relevant documentation:
    # https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-authentication.html#cli-chap-authentication-precedence

    variables = [
        (
            "Spacer config - AWS_PROFILE_NAME (highest priority if specified)",
            config.AWS_PROFILE_NAME,
        ),
        (
            "Environment var - AWS_CONFIG_FILE",
            os.getenv('AWS_CONFIG_FILE'),
        ),
        (
            "Environment var - AWS_SHARED_CREDENTIALS_FILE",
            os.getenv('AWS_SHARED_CREDENTIALS_FILE'),
        ),
        (
            "Spacer config - AWS_ACCESS_KEY_ID",
            config.AWS_ACCESS_KEY_ID,
        ),
        (
            "Spacer config - AWS_SECRET_ACCESS_KEY",
            config.AWS_SECRET_ACCESS_KEY,
        ),
        (
            "Spacer config - AWS_SESSION_TOKEN",
            config.AWS_SESSION_TOKEN,
        ),
    ]

    for description, value in variables:
        value_status = "Yes" if value else "No"
        print(f"{description}: {value_status}")

    # Despite above observations, we deliberately do not try to say whether
    # or why a session will be successfully made or not. We just print the
    # relevant variables we know of (above) and then actually try to get a
    # session, letting AWS's error messages attempt to elucidate their own
    # complex auth logic.

    print("Will now call create_aws_session()...")

    # If this gets an error, just let the error happen.
    session = create_aws_session()

    print("Will now call get_caller_identity()...")

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sts/client/get_caller_identity.html
    # Again, if this gets an error, just let the error happen.
    response = session.client('sts').get_caller_identity()

    # ARN can give a bit more insight on the type of auth that worked.
    # For example:
    # - If we used an EC2 instance role to automatically get credentials,
    #   the ARN should contain an instance ID like i-<hex digits>.
    # - If we used a config file with IAM Roles Anywhere, the end of the ARN
    #   should have 40 hex digits.
    # - If we got temporary credentials from AWS CLI, then the
    #   session name given in that command should be in the ARN.
    # - If we got temporary credentials from SSO, then the SSO username
    #   like <name>@UCSD.EDU should be in the ARN.
    print(f"Identified as ARN: {response['Arn']}")
