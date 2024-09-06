from __future__ import annotations
import json
import time
from collections import defaultdict
from datetime import datetime
from logging import getLogger

import boto3
import numpy as np
from botocore.errorfactory import ClientError

from spacer import config
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg
from spacer.messages import JobReturnMsg
from spacer.storage import store_image, load_image

logger = getLogger(__name__)


def aws_batch_submit(job_queue: str,
                     job_msg_loc: DataLocation,
                     job_res_loc: DataLocation):
    """ Submits job to AWS batch """

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


def aws_batch_job_status(jobs: list[tuple[str, DataLocation, JobMsg,
                                          DataLocation, DataLocation, int]]):
    """ Monitor AWS batch status.
    Input should be tuple same as the output of submit_jobs
    :type jobs: object
    """
    state = defaultdict(int)
    runtimes = defaultdict(float)

    for job_id, feat_loc, _, _, job_res_loc, _ in jobs:
        client = boto3.client('batch')
        resp = client.describe_jobs(jobs=[job_id])
        assert len(resp['jobs']) == 1
        job_status = resp['jobs'][0]['status']
        state[job_status] += 1

        if job_status == 'SUCCEEDED' and feat_loc is not None:

            # Double check that the out_key is actually there.
            s3 = config.get_s3_resource()
            try:
                s3.Object(config.TEST_BUCKET, feat_loc.key).load()
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logger.info(
                        "JOB: {} marked as SUCCEEDED, but missing key at {}".
                            format(job_id, feat_loc.key)
                    )
                else:
                    logger.error(
                        "Something else is wrong: {} {}".format(job_id, str(e))
                    )

            # Load results and read out the runtime.
            try:
                job_res = JobReturnMsg.load(job_res_loc)
                runtimes[job_id] = job_res.results[0].runtime
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logger.info(
                        "JOB: {} marked as SUCCEEDED, but missing key at {}".
                            format(job_id, job_res_loc.key)
                    )
                else:
                    logger.error(
                        "Something else is wrong: {} {}".format(job_id, str(e))
                    )

        if job_status == 'FAILED':
            logger.info('JOB: {} failed!'.format(job_id))

    return state, runtimes


def submit_jobs(nbr_rowcols: list[int],
                job_queue: str = 'shakeout',
                image_size: int = 1000,
                extractor_name: str = 'efficientnet_b0_ver1'):
    """ Creates and submits jobs to AWS Batch. """
    assert max(nbr_rowcols) <= 1000
    assert max(nbr_rowcols) <= image_size

    logger.info('Submitting {} jobs... '.format(nbr_rowcols))
    targets = []

    # Load up an old image and resize it to desired size.
    org_img_loc = DataLocation(storage_type='s3',
                               key='08bfc10v7t.png',
                               bucket_name=config.TEST_BUCKET)
    org_img = load_image(org_img_loc)

    img = org_img.resize((image_size, image_size)).convert("RGB")

    img_loc = DataLocation(storage_type='s3',
                           key='tmp/{}.jpg'.
                           format(str(datetime.now()).replace(' ', '_')),
                           bucket_name=config.TEST_BUCKET)
    store_image(img_loc, img)
    for npts in nbr_rowcols:

        # Assign each job a unique feature target location, so we can monitor
        # as the jobs are completed. Use timestamp to get a unique name.
        feat_loc = DataLocation(storage_type='s3',
                                key='tmp/{}.feats.json'.
                                format(str(datetime.now()).replace(' ', '_')),
                                bucket_name=config.TEST_BUCKET)
        job_msg = JobMsg(
            task_name='extract_features',
            tasks=[ExtractFeaturesMsg(
                job_token='regression_job',
                feature_extractor_name=extractor_name,
                rowcols=[(i, i) for i in list(range(npts))],
                image_loc=img_loc,
                feature_loc=feat_loc
            )])

        job_msg_loc = DataLocation(
            storage_type='s3',
            key=feat_loc.key + '.job_msg.json',
            bucket_name=config.TEST_BUCKET
        )
        job_res_loc = DataLocation(
            storage_type='s3',
            key=feat_loc.key + '.job_res.json',
            bucket_name=config.TEST_BUCKET
        )
        job_msg.store(job_msg_loc)

        job_id = aws_batch_submit(job_queue, job_msg_loc, job_res_loc)

        targets.append((job_id, feat_loc, job_msg, job_msg_loc, job_res_loc,
                        image_size))

    logger.info('{} jobs submitted.'.format(len(targets)))
    return targets


def monitor_jobs(targets):
    """ Monitor jobs. Input should be same as output of submit_jobs. """
    status = {'SUCCEEDED': 0}
    while status['SUCCEEDED'] < len(targets):
        status, runtimes = aws_batch_job_status(targets)
        logger.info('Job status: {}, mean runtime: {:.2f} seconds.'.format(
            dict(status), np.mean(list(runtimes.values()))))
        time.sleep(3)
    logger.info("All jobs done.")
    return status, runtimes