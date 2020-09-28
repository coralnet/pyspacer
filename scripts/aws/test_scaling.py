"""
This script submits 100 jobs to queues and monitors as the
jobs are completed. The cluster is setup to add 20 instances as soon as there
are jobs in the test_queue, so the 100 jobs are be completed quickly.
"""

import json
import logging
import time
from datetime import datetime

from scripts.aws.utils import count_jobs_complete, aws_batch_submit, \
    aws_batch_queue_status, sqs_purge
from spacer import config
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg, \
    JobReturnMsg


def submit_jobs(job_cnt, job_queue, results_queue, extractor_name):
    """ Submits job_cnt jobs. """

    print('-> Submitting {} jobs... '.format(job_cnt))
    targets = []
    for _ in range(job_cnt):

        # Assign each job a unique feature target location, so we can monitor
        # as the jobs are completed. Use timestamp to get a unique name.
        feat_loc = DataLocation(storage_type='s3',
                                key='tmp/08bfc10v7t.png.{}.feats.json'.
                                format(str(datetime.now()).replace(' ', '_')),
                                bucket_name='spacer-test')
        msg = JobMsg(
            task_name='extract_features',
            tasks=[ExtractFeaturesMsg(
                job_token='regression_job',
                feature_extractor_name=extractor_name,
                rowcols=[(10, 10)],
                image_loc=DataLocation(storage_type='s3',
                                       key='08bfc10v7t.png',
                                       bucket_name='spacer-test'),
                feature_loc=feat_loc
            )])

        job_msg_loc = DataLocation(
            storage_type='s3',
            key=feat_loc.key + '.job_msg.json',
            bucket_name='spacer-test'
        )
        msg.store(job_msg_loc)
        aws_batch_submit(job_queue, results_queue, job_msg_loc)
        targets.append(feat_loc)

    logging.info('{} jobs submitted.'.format(len(targets)))
    return targets


def purge_and_calc_runtime(queue_name):

    conn = config.get_sqs_conn()
    queue = conn.get_queue(queue_name)
    m = queue.read()
    runtimes = []
    while m is not None:
        return_msg = JobReturnMsg.deserialize(json.loads(m.get_body()))
        assert return_msg.ok
        runtimes.append(return_msg.results[0].runtime)
        queue.delete_message(m)
        m = queue.read()

    print('-> Purged {} messages from {}'.format(len(runtimes), queue_name))
    if len(runtimes) > 0:
        return sum(runtimes) / len(runtimes)
    else:
        return 0


def main(job_queue='shakeout',
         results_queue='spacer_shakeout_results',
         extractor_name='efficientnet_b0_ver1'):

    logging.info("-> Starting scaling test for {}.".format(extractor_name))
    sqs_purge(results_queue)

    targets = submit_jobs(100, job_queue, results_queue, extractor_name)
    complete_count = 0
    base = aws_batch_queue_status(job_queue)
    while complete_count < len(targets):
        logging.info(aws_batch_queue_status(job_queue, base))
        complete_count = count_jobs_complete(targets)
        logging.info('Jobs complete: {}'.format(complete_count))
        time.sleep(3)

    logging.info("-> All jobs done.")
    runtime = purge_and_calc_runtime(results_queue)
    logging.info("-> Average runtime: {}".format(runtime))


if __name__ == '__main__':
    main()
