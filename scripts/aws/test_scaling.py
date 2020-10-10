"""
This script submits 100 jobs to queues and monitors as the
jobs are completed. The cluster is setup to add 20 instances as soon as there
are jobs in the test_queue, so the 100 jobs are be completed quickly.
"""


import logging
import time
from datetime import datetime

import numpy as np

from scripts.aws.utils import aws_batch_submit, \
    aws_batch_job_status
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg


def submit_jobs(job_cnt, job_queue, extractor_name):
    """ Submits job_cnt jobs. """

    logging.info('Submitting {} jobs... '.format(job_cnt))
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
        job_res_loc = DataLocation(
            storage_type='s3',
            key=feat_loc.key + '.job_res.json',
            bucket_name='spacer-test'
        )
        msg.store(job_msg_loc)
        job_id = aws_batch_submit(job_queue, job_msg_loc, job_res_loc)
        targets.append((job_id, feat_loc, job_res_loc))

    logging.info('{} jobs submitted.'.format(len(targets)))
    return targets


def main(job_queue='shakeout',
         extractor_name='efficientnet_b0_ver1'):

    logging.info("Starting scaling test for {}.".format(extractor_name))
    targets = submit_jobs(100, job_queue, extractor_name)

    status = {'SUCCEEDED': 0}
    while status['SUCCEEDED'] < len(targets):
        status, runtimes = aws_batch_job_status(targets)
        logging.info(status)
        logging.info("Avg runtime: {.2f)".format(np.mean(runtimes)))
        time.sleep(3)
    logging.info("All jobs done.")


if __name__ == '__main__':
    main()
