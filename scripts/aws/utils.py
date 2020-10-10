import json
import logging
import boto3
from botocore.errorfactory import ClientError
from typing import List, Tuple
from collections import defaultdict
from spacer import config
from spacer.messages import JobReturnMsg, DataLocation


def aws_batch_submit(job_queue, results_queue, job_msg_loc: DataLocation):

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
                    'name': 'OUT_QUEUE',
                    'value': results_queue,
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
                s3.Object('spacer-test', feat_loc.key).load()
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
                        format(job_id, job_res.key)
                    )
                else:
                    logging.error(
                        "Something else is wrong: {} {}".format(job_id, str(e))
                    )

        if job_status == 'FAILED':
            logging.info('JOB: {} failed!'.format(job_id))

    return state, runtimes


def sqs_purge(queue_name):
    """ Deletes all messages in queue. """

    conn = config.get_sqs_conn()
    queue = conn.get_queue(queue_name)
    m = queue.read()
    count = 0
    while m is not None:
        queue.delete_message(m)
        m = queue.read()
        count += 1
    print('Purged {} messages from {}'.format(count, queue_name))


def sqs_fetch(queue_name):

    conn = config.get_sqs_conn()
    queue = conn.get_queue(queue_name)
    m = queue.read()
    job_cnt = 0
    while m is not None:
        body = m.get_body()
        print("return message body:", body)
        return_msg = JobReturnMsg.deserialize(json.loads(m.get_body()))
        job_token = return_msg.original_job.tasks[0].job_token
        if return_msg.ok:
            logging.info('{} done in {:.2f} s.'.format(
                job_token, return_msg.results[0].runtime))
        else:
            logging.info('{} failed with: {}.'.format(
                job_token, return_msg.error_message))
        queue.delete_message(m)
        m = queue.read()
        job_cnt += 1
    return job_cnt
