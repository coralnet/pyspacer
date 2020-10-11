import json
import logging
import boto3
from botocore.errorfactory import ClientError
from typing import List, Tuple
from collections import defaultdict
from spacer import config
from spacer.messages import JobReturnMsg, DataLocation


def aws_batch_submit(job_queue: str,
                     job_msg_loc: DataLocation,
                     job_res_loc: DataLocation):

    client = boto3.client('batch')
    resp = client.submit_job(
        jobQueue=job_queue,
        jobName='test_job_from_spacer',
        jobDefinition='spacer-job',
        containerOverrides={
            'environment': [
                {
                    'name': 'JOB_MSG_LOC',
                    'value': json.dumps(job_msg_loc.serialize()),
                },
                {
                    'name': 'RES_MSG_LOC',
                    'value': json.dumps(job_res_loc.serialize()),
                },
            ],
        }
    )
    return resp['jobId']


def aws_batch_job_status(jobs: List[Tuple[str, DataLocation, DataLocation]]):
    """ Input should be tuple of
    (AWE Batch job_id,
     a DataLocation to where we expect something to be written,
     a DataLocation with serialized JobRes message)
     The second entry is used as a sanity check and is ignored if None.
     The third entry is not used in this function
    """
    state = defaultdict(int)
    runtimes = []

    for job_id, feat_loc, res_loc in jobs:
        client = boto3.client('batch')
        resp = client.describe_jobs(jobs=[job_id])
        assert len(resp['jobs']) == 1
        job_status = resp['jobs'][0]['status']
        state[job_status] += 1

        if job_status == 'SUCCEEDED' and feat_loc is not None:

            # Double check that the out_key is actually there.
            s3 = config.get_s3_conn()
            try:
                s3.Object(config.TEST_BUCKET, feat_loc.key).load()
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logging.info(
                        "JOB: {} marked as SUCCEEDED, but missing key at {}".
                        format(job_id, feat_loc.key)
                    )
                else:
                    logging.error(
                        "Something else is wrong: {} {}".format(job_id, str(e))
                    )

            # Load results and read out the runtime.
            try:
                job_res = JobReturnMsg.load(res_loc)
                runtimes.append(job_res.results[0].runtime)
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logging.info(
                        "JOB: {} marked as SUCCEEDED, but missing key at {}".
                        format(job_id, res_loc.key)
                    )
                else:
                    logging.error(
                        "Something else is wrong: {} {}".format(job_id, str(e))
                    )

        if job_status == 'FAILED':
            logging.info('JOB: {} failed!'.format(job_id))

    return state, runtimes
