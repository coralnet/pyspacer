"""
This script submits a few jobs to our AWS cluster. The goal is to test
1) how large images we can extract features for before running out of memory.
2) how many patches we can handle.
If there is a difference across feature extractors, this needs to be set to the
min across the extractors, and then get encoded in the config file.
"""

from datetime import datetime

import fire
from PIL import Image

from spacer.mailman import process_job
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg
from spacer.storage import store_image

IMAGE_SIZES = [
    (1000, 1000),  # 1 mega pixel
    (3000, 3000),  # 10 mega pixel
    (10000, 10000),  # 100 mega pixel
    (30000, 30000),  # 1000 mega pixel
    (50000, 50000),  # 2500 mega pixel
]


def run_jobs(extractor_name):
    for (nrows, ncols) in IMAGE_SIZES:
        log('Running job: ({}, {})'.format(nrows, ncols))

        # Create a image and upload to s3.
        img = Image.new('RGB', (nrows, ncols))
        img_loc = DataLocation(storage_type='filesystem',
                               key='tmp.{}_{}.jpg'.format(nrows, ncols))
        store_image(img_loc, img)
        del img

        feat_loc = DataLocation(storage_type='filesystem',
                                key=img_loc.key + '.feats')
        msg = JobMsg(
            task_name='extract_features',
            tasks=[ExtractFeaturesMsg(
                job_token='({}, {})'.format(nrows, ncols),
                feature_extractor_name=extractor_name,
                rowcols=[(1, 1)],
                image_loc=img_loc,
                feature_loc=feat_loc
            )])

        return_msg = process_job(msg)
        log(str(return_msg))


def log(msg):
    msg = '['+datetime.now().strftime("%H:%M:%S") + '] ' + msg
    with open('memory_test.log', 'a') as f:
        f.write(msg + '\n')
        print(msg)


def main(extractor_name='efficientnet_b0_ver1'):

    log("Testing memory local {}.".format(extractor_name))
    run_jobs(extractor_name)


if __name__ == '__main__':
    fire.Fire()
