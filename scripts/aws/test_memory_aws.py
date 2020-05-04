"""
This script submits a few jobs to our AWS cluster. The goal is to test
1) how large images we can extract features for before running out of memory.
2) how many patches we can handle.
If there is a difference across feature extractors, this needs to be set to the
min across the extractors, and then get encoded in the config file.
"""

import json
import time
from datetime import datetime

import fire
from PIL import Image

from scripts.aws.utils import sqs_status, purge
from spacer import config
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg, \
    JobReturnMsg
from spacer.storage import store_image

IMAGE_SIZES = [
    (5000, 5000),  # 10 mega pixel
    (10000, 10000),  # 100 mega pixel
    (15000, 15000),  # 225 mega pixel
]

NBR_ROWCOLS = [100, 1000, 3000, 5000]


def submit_jobs(queue_name):

    log('Submitting memory test jobs to {}.'.format(queue_name))

    targets = []
    for extractor_name in ['vgg16_coralnet_ver1', 'efficientnet_b0_ver1']:

        for (nrows, ncols) in IMAGE_SIZES:

            # Create a image and upload to s3.
            img = Image.new('RGB', (nrows, ncols))
            img_loc = DataLocation(storage_type='s3',
                                   key='tmp/{}_{}.jpg'.format(nrows, ncols),
                                   bucket_name='spacer-test')
            store_image(img_loc, img)

            for npts in NBR_ROWCOLS:

                feat_loc = DataLocation(storage_type='s3',
                                        key=img_loc.key + '.{}feats'.format(npts),
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
                conn = config.get_sqs_conn()
                in_queue = conn.get_queue(queue_name)
                msg = in_queue.new_message(body=json.dumps(msg.serialize()))
                in_queue.write(msg)
                targets.append(feat_loc)

    log('{} jobs submitted.'.format(len(targets)))
    return targets


def log(msg):
    msg = '['+datetime.now().strftime("%H:%M:%S") + '] ' + msg
    with open('memory_test.log', 'a') as f:
        f.write(msg + '\n')
        print(msg)


def fetch_jobs(queue_name):

    conn = config.get_sqs_conn()
    queue = conn.get_queue(queue_name)
    m = queue.read()
    job_cnt = 0
    while m is not None:
        return_msg = JobReturnMsg.deserialize(json.loads(m.get_body()))
        job_token = return_msg.original_job.tasks[0].job_token
        if return_msg.ok:
            log('{} done in {:.2f} s.'.format(job_token,
                                              return_msg.results[0].runtime))
        else:
            log('{} failed with: {}.'.format(job_token,
                                             return_msg.error_message))
        queue.delete_message(m)
        m = queue.read()
        job_cnt += 1
    return job_cnt


def main(jobs_queue='spacer_test_jobs',
         results_queue='spacer_test_results'):

    log("Starting ECS feature extraction.")
    purge(jobs_queue)
    purge(results_queue)

    _ = submit_jobs(jobs_queue)
    complete_count = 0
    while True:
        jobs_todo, jobs_ongoing = sqs_status(jobs_queue)
        results_todo, _ = sqs_status(results_queue)
        complete_count += fetch_jobs(results_queue)
        log("Status: {} todo, {} ongoing, {} in results queue {} done".format(
            jobs_todo, jobs_ongoing, results_todo, complete_count))
        time.sleep(60)


if __name__ == '__main__':
    fire.Fire()
