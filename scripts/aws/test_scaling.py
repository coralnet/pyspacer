"""
This script submits 100 jobs to queues and monitors as the
jobs are completed. The cluster is setup to add 20 instances as soon as there
are jobs in the test_queue, so the 100 jobs are be completed quickly.
"""

import json
import logging
import time
from datetime import datetime

from scripts.aws.utils import aws_batch_submit, \
    sqs_purge, aws_batch_job_status
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
        job_id = aws_batch_submit(job_queue, results_queue, job_msg_loc)
        targets.append((job_id, feat_loc.key))

    logging.info('{} jobs submitted.'.format(len(targets)))
    return targets


def main(job_queue='shakeout',
         results_queue='spacer_shakeout_results',
         extractor_name='efficientnet_b0_ver1'):

    logging.info("-> Starting scaling test for {}.".format(extractor_name))
    sqs_purge(results_queue)
    targets = submit_jobs(100, job_queue, results_queue, extractor_name)

    status = {'SUCCEEDED': 0}
    while status['SUCCEEDED'] < len(targets):
        status = aws_batch_job_status(targets)
        logging.info(status)
        time.sleep(3)
    logging.info("-> All jobs done.")


if __name__ == '__main__':
    main()
