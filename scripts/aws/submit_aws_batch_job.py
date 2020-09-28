import logging
import time

from scripts.aws.utils import \
    sqs_purge, aws_batch_queue_status, sqs_fetch, aws_batch_submit
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg


def submit(job_queue, results_queue):

    extract_task = ExtractFeaturesMsg(
        job_token='regression_job',
        feature_extractor_name='efficientnet_b0_ver1',
        rowcols=[(20, 265),
                 (76, 295),
                 (59, 274),
                 (151, 62)],
        image_loc=DataLocation(storage_type='s3',
                               key='08bfc10v7t.png',
                               bucket_name='spacer-test'),
        feature_loc=DataLocation(storage_type='s3',
                                 key='tmp/08bfc10v7t.feats.json',
                                 bucket_name='spacer-test'))

    msg = JobMsg(
        task_name='extract_features',
        tasks=[extract_task]
    )

    job_msg_loc = DataLocation(
        storage_type='s3',
        key='tmp_job.json',
        bucket_name='spacer-test'
    )
    msg.store(job_msg_loc)

    aws_batch_submit(job_queue, results_queue, job_msg_loc)


def main(job_queue='shakeout', results_queue='spacer_shakeout_results'):

    logging.info("Starting batch job shakeout.")
    sqs_purge(results_queue)

    submit(job_queue, results_queue)
    res_cnt = 0
    while res_cnt == 0:
        print(aws_batch_queue_status(job_queue))
        res_cnt = sqs_fetch(results_queue)
        time.sleep(3)


if __name__ == '__main__':
    main()
