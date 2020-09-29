import json
import logging
import boto3
from collections import defaultdict
from spacer import config
from spacer.messages import JobReturnMsg, DataLocation

BATCH_STATUS_NAMES = [
        'SUBMITTED',
        'PENDING',
        'RUNNABLE',
        'STARTING',
        'RUNNING',
        'SUCCEEDED',
        'FAILED'
    ]


def aws_batch_queue_status(queue_name, base=None):

    client = boto3.client('batch')
    report = {
        status: len(client.list_jobs(
            jobQueue=queue_name,
            jobStatus=status
        )['jobSummaryList']) for status in BATCH_STATUS_NAMES
    }
    if base is not None:
        for status in BATCH_STATUS_NAMES:
            report[status] -= base[status]
    return report


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


def aws_batch_job_status(job_ids):
    state = defaultdict(int)

    for job_id in job_ids:
        client = boto3.client('batch')
        resp = client.describe_jobs(jobs=[job_id])
        assert len(resp['jobs']) == 1
        state[resp['jobs'][0]['status']] += 1
    return state


def count_jobs_complete(targets):
    """ Check the target locations and counts how many are complete. """

    conn = config.get_s3_conn()
    bucket = conn.get_bucket('spacer-test', validate=True)

    complete_count = 0
    for target, job_id in targets:
        key = bucket.get_key(target.key)
        if key is not None:
            complete_count += 1
        else:
            logging.info("job id: {} not complete".format(job_id))

    return complete_count


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
    print('-> Purged {} messages from {}'.format(count, queue_name))


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
