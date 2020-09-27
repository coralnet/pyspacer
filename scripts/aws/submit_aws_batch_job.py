import boto3
import json
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg
from scripts.aws.utils import purge, fetch_jobs
import logging


def submit_to_batch(results_queue):

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

    client = boto3.client('batch')
    client.submit_job(
        jobQueue='production',
        jobName='test_job_from_spacer',
        jobDefinition='spacer-job',
        containerOverrides={
            'environment': [
                {
                    'name': 'JOB_MSG',
                    'value': json.dumps(msg.serialize()),
                },
                {
                    'name': 'OUT_QUEUE',
                    'value': results_queue,
                },
            ],
        }
    )


def main(results_queue='spacer_test_results'):

    logging.info("Starting batch job shakeout.")
    purge(results_queue)

    submit_to_batch(results_queue)
    fetch_jobs(results_queue)


if __name__ == '__main__':
    main()
