"""
This script submits 100 jobs to queues and monitors as the
jobs are completed. The cluster is setup to add 20 instances as soon as there
are jobs in the test_queue, so the 100 jobs are be completed quickly.
"""

import json
import time
import fire
from datetime import datetime

from spacer import config
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg, JobReturnMsg


def submit_jobs(job_cnt, queue_name, extractor_name):
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
        conn = config.get_sqs_conn()
        in_queue = conn.get_queue(queue_name)
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


def purge_and_calc_runtime(queue_name):

    conn = config.get_sqs_conn()
    queue = conn.get_queue(queue_name)
    m = queue.read()
    runtimes = []
    while m is not None:
        return_msg = JobReturnMsg.deserialize(json.loads(m.get_body()))
        assert return_msg.ok
        runtimes.append(return_msg.results[0].runtime)
        queue.delete_message(m)
        m = queue.read()

    print('-> Purged {} messages from {}'.format(len(runtimes), queue_name))
    if len(runtimes) > 0:
        return sum(runtimes) / len(runtimes)
    else:
        return 0


def purge(queue_name):
    """ Deletes all messages in queue. """

    conn = config.get_sqs_conn()
    queue = conn.get_queue(queue_name)
    m = queue.read()
    count = 0
    while m is not None:
        m = queue.read()
        count += 1
    print('-> Purged {} messages from {}'.format(count, queue_name))


def main(jobs_queue='spacer_test_jobs',
         results_queue='spacer_test_results',
         extractor_name='efficientnet_b0_ver1'):

    print("-> Starting ECS feature extraction for {}.".format(extractor_name))
    purge(jobs_queue)
    purge(results_queue)

    targets = submit_jobs(100, jobs_queue, extractor_name)
    complete_count = 0
    while complete_count < len(targets):
        jobs_todo, jobs_ongoing = sqs_status(jobs_queue)
        results_todo, _ = sqs_status(results_queue)
        complete_count = count_jobs_complete(targets)
        print("-> [{}] Status: {} todo, {} ongoing, {} in results queue, "
              "{} done".format(datetime.now().strftime("%H:%M:%S"),
                               jobs_todo, jobs_ongoing, results_todo,
                               complete_count))
        time.sleep(10)

    print("-> All jobs done.")
    runtime = purge_and_calc_runtime(results_queue)
    print("-> Average runtime: {}".format(runtime))


if __name__ == '__main__':
    fire.Fire()
