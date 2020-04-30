from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg
from spacer import config
import json


def submit_simple(job_cnt):
    for jobid in range(job_cnt):
        msg = JobMsg(
            task_name='extract_features',
            tasks=[ExtractFeaturesMsg(
                job_token='regression_job',
                feature_extractor_name='vgg16_coralnet_ver1',
                rowcols=[(10, 10)],
                image_loc=DataLocation(storage_type='s3',
                                       key='08bfc10v7t.png',
                                       bucket_name='spacer-test'),
                feature_loc=DataLocation(storage_type='s3',
                                         key='tmp/08bfc10v7t.png.feats{}.json'.
                                         format(jobid),
                                         bucket_name='spacer-test')
            )])

        queue_name = 'spacer_test_jobs'
        conn = config.get_sqs_conn()
        in_queue = conn.get_queue(queue_name)
        msg = in_queue.new_message(body=json.dumps(msg.serialize()))
        in_queue.write(msg)
        print('submitted job {}'.format(jobid))


def read_results():

    queue_name = 'spacer_test_results'
    conn = config.get_sqs_conn()
    queue = conn.get_queue(queue_name)
    # Read message
    m = queue.read()
    while m is not None:
        print(m.get_body())
        queue.delete_message(m)
        m = queue.read()

    print('Done reading results queue.')


if __name__ == '__main__':
    read_results()
    submit_simple(5)
