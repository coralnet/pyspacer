"""
This script submits a few jobs to our AWS cluster. The goal is to test
1) how large images we can extract features for before running out of memory.
2) how many patches we can handle.
If there is a difference across feature extractors, this needs to be set to the
min across the extractors, and then get encoded in the config file.
"""

import logging
import time

from PIL import Image

from scripts.aws.utils import sqs_purge, aws_batch_job_status, aws_batch_submit
from spacer import config
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg
from spacer.storage import store_image

IMAGE_SIZES = [
    (5000, 5000),  # 10 mega pixel
    (10000, 10000),  # 100 mega pixel
]

NBR_ROWCOLS = [10, 100, 200, 1000]


def submit_jobs(job_queue, results_queue):

    logging.info('Submitting memory test jobs to {}.'.format(job_queue))

    targets = []
    for (nrows, ncols) in IMAGE_SIZES:

        # Create a image and upload to s3.
        img = Image.new('RGB', (nrows, ncols))
        img_loc = DataLocation(storage_type='s3',
                               key='tmp/{}_{}.jpg'.format(nrows, ncols),
                               bucket_name=config.TEST_BUCKET)
        store_image(img_loc, img)

        for extractor_name in config.FEATURE_EXTRACTOR_NAMES:

            for npts in NBR_ROWCOLS:
                feat_key = img_loc.key + '.{}{}.feats.json'. \
                    format(npts, extractor_name)
                feat_loc = DataLocation(storage_type='s3',
                                        key=feat_key,
                                        bucket_name=config.TEST_BUCKET)
                msg = JobMsg(
                    task_name='extract_features',
                    tasks=[ExtractFeaturesMsg(
                        job_token='{} ({}, {}): {}'.format(
                            extractor_name, nrows, ncols, npts),
                        feature_extractor_name=extractor_name,
                        rowcols=[(i, i) for i in list(range(npts))],
                        image_loc=img_loc,
                        feature_loc=feat_loc
                    )])

                job_msg_loc = DataLocation(
                    storage_type='s3',
                    key=feat_key + '.job_msg.json',
                    bucket_name=config.TEST_BUCKET
                )
                msg.store(job_msg_loc)
                job_id = aws_batch_submit(job_queue, results_queue, job_msg_loc)
                targets.append((job_id, feat_loc.key))

    logging.info('{} jobs submitted.'.format(len(targets)))
    return targets


def main(job_queue='shakeout',
         results_queue='spacer_shakeout_results'):

    logging.info("Starting ECS feature extraction.")
    sqs_purge(results_queue)
    targets = submit_jobs(job_queue, results_queue)
    status = {'SUCCEEDED': 0}
    while status['SUCCEEDED'] < len(targets):
        status = aws_batch_job_status(targets)
        logging.info(status)
        time.sleep(3)
    logging.info("All jobs done.")


if __name__ == '__main__':
    main()
