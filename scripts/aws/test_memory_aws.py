"""
This script submits a few jobs to our AWS cluster. The goal is to test
1) how large images we can extract features for before running out of memory.
2) how many patches we can handle.
If there is a difference across feature extractors, this needs to be set to the
min across the extractors, and then get encoded in the config file.
"""

import json
import time
import boto3
from datetime import datetime

import fire
from PIL import Image

from scripts.aws.utils import sqs_status, purge, fetch_jobs, submit_to_batch, \
    batch_queue_status
from spacer import config
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg
from spacer.storage import store_image

IMAGE_SIZES = [
    (5000, 5000),  # 10 mega pixel
    (10000, 10000),  # 100 mega pixel
    # (15000, 15000),  # 225 mega pixel
]

NBR_ROWCOLS = [10, 100, 200, 1000]


def submit_jobs(job_queue, results_queue):

    log('Submitting memory test jobs to {}.'.format(job_queue))

    targets = []
    for (nrows, ncols) in IMAGE_SIZES:

        # Create a image and upload to s3.
        img = Image.new('RGB', (nrows, ncols))
        img_loc = DataLocation(storage_type='s3',
                               key='tmp/{}_{}.jpg'.format(nrows, ncols),
                               bucket_name='spacer-test')
        store_image(img_loc, img)

        for extractor_name in config.FEATURE_EXTRACTOR_NAMES:

            for npts in NBR_ROWCOLS:
                feat_key = img_loc.key + '.{}{}.feats.json'.\
                    format(npts, extractor_name)
                feat_loc = DataLocation(storage_type='s3',
                                        key=feat_key,
                                        bucket_name='spacer-test')
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
                    bucket_name='spacer-test'
                )
                msg.store(job_msg_loc)

                submit_to_batch(job_queue, results_queue, job_msg_loc)
                targets.append(feat_loc)

    log('{} jobs submitted.'.format(len(targets)))
    return targets


def log(msg):
    msg = '['+datetime.now().strftime("%H:%M:%S") + '] ' + msg
    with open('memory_test.log', 'a') as f:
        f.write(msg + '\n')
        print(msg)


def main(job_queue='shakeout',
         results_queue='spacer_test_results'):

    log("Starting ECS feature extraction.")
    purge(results_queue)
    base = batch_queue_status(job_queue)
    print(base)
    _ = submit_jobs(job_queue, results_queue)
    complete_count = 0
    while True:
        print(batch_queue_status(job_queue, base))
        complete_count += fetch_jobs(results_queue)
        log("{} complete".format(complete_count))
        time.sleep(5)


if __name__ == '__main__':
    main()
