"""
This script submits 100 jobs to spacer_test_jobs queue and monitors as the
jobs are completed. The cluster is setup to add 20 instances as soon as there
are jobs in the test_queue, so the 100 jobs are be completed quickly.
"""

import json
import time
from datetime import datetime

from spacer import config
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg


def submit_jobs(job_cnt):
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
                feature_extractor_name='vgg16_coralnet_ver1',
                rowcols=[(10, 10)],
                image_loc=DataLocation(storage_type='s3',
                                       key='08bfc10v7t.png',
                                       bucket_name='spacer-test'),
                feature_loc=feat_loc
            )])
        conn = config.get_sqs_conn()
        in_queue = conn.get_queue('spacer_test_jobs')
        msg = in_queue.new_message(body=json.dumps(msg.serialize()))
        in_queue.write(msg)
        targets.append(feat_loc)

    print('-> {} jobs submitted.'.format(job_cnt))
    return targets


def sqs_status(queue_name):
    """
    Returns number of pending and ongoing jobs in the queue.
    """
    conn = config.get_sqs_conn()
    queue = conn.get_queue(queue_name)
    attr = queue.get_attributes()
    return int(attr['ApproximateNumberOfMessages']), \
           int(attr['ApproximateNumberOfMessagesNotVisible'])


def count_jobs_complete(targets):
    """ Check the target locations and counts how many are complete. """

    conn = config.get_s3_conn()
    bucket = conn.get_bucket('spacer-test', validate=True)

    complete_count = 0
    for target in targets:
        key = bucket.get_key(target.key)
        if key is not None:
            complete_count += 1

    return complete_count


def purge_results(queue_name):
    """ Deletes all messages in queue. """

    conn = config.get_sqs_conn()
    queue = conn.get_queue(queue_name)
    m = queue.read()
    count = 0
    while m is not None:
        queue.delete_message(m)
        m = queue.read()
        count += 1

    print('-> Purged {} messages from {}'.format(count, queue_name))


def main():
    print("-> Starting ECS shakeout script.")
    purge_results('spacer_test_results')
    purge_results('spacer_test_jobs')

    targets = submit_jobs(100)
    complete_count = 0
    while complete_count < len(targets):
        jobs_todo, jobs_ongoing = sqs_status('spacer_test_jobs')
        results_todo, _ = sqs_status('spacer_test_results')
        complete_count = count_jobs_complete(targets)
        print("-> [{}] Status: {} todo, {} ongoing, {} done, {} extracted".
              format(datetime.now().strftime("%H:%M:%S"),
                     jobs_todo, jobs_ongoing, results_todo, complete_count))
        time.sleep(10)
    print("-> All jobs done, purging results queue")
    purge_results('spacer_test_results')


if __name__ == '__main__':
    main()
