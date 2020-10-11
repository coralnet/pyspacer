"""
Defines the highest-level method for task handling through AWS SQS.
"""

import json
import logging
import os

import fire

from spacer import config
from spacer.messages import JobMsg, DataLocation
from spacer.tasks import \
    process_job


def env_job(): # pragma: no cover
    """ Runs a job defined in environmental variables. Setup to play
    nicely with AWS Batch.
    Expects JOB_MSG_LOC to contain the json-serialized DataLocation of JobMgs.
    Expects OUT_QUEUE to contain the name of the desired results queue.
    """

    job_msg_loc = os.getenv('JOB_MSG_LOC')
    if job_msg_loc is None:
        raise ValueError('JOB_MSG_LOC env. variable not set. '
                         'Can not process job.')

    logging.info(" Received job for ENV {}.".format(job_msg_loc))

    with config.log_entry_and_exit('job message location deserialization'):
        job_msg_loc = DataLocation.deserialize(json.loads(job_msg_loc))

    with config.log_entry_and_exit('job message initialization'):
        job_msg = JobMsg.load(job_msg_loc)

    try:
        job_return_msg = process_job(job_msg)
        job_return_msg_dict = job_return_msg.serialize()
    except Exception as e:
        # Handle deserialization errors directly in mailman.
        # All other errors are handled in "handle_message" function.
        job_return_msg_dict = {
            'original_job': job_msg.serialize(),
            'ok': False,
            'results': None,
            'error_message': 'Error deserializing message: ' + repr(e)
        }

    # Return
    out_queue_name = os.getenv('RES_SQS_QUEUE')
    if out_queue_name is not None:
        with config.log_entry_and_exit('writing results to {}'.format(
                out_queue_name)):
            conn = config.get_sqs_conn()
            out_queue = conn.get_queue(out_queue_name)
            m_out = out_queue.new_message(body=json.dumps(job_return_msg_dict))
            out_queue.write(m_out)

    out_msg_loc = os.getenv('RES_MSG_LOC')
    if out_msg_loc is not None:
        with config.log_entry_and_exit('out message location deserialization'):
            out_msg_loc = DataLocation.deserialize(json.loads(out_msg_loc))

        with config.log_entry_and_exit('writing res to {}'.format(
                out_msg_loc.key)):
            s3 = config.get_s3_conn()
            s3.Bucket(out_msg_loc.bucket_name).put_object(
                Key=out_msg_loc.key,
                Body=bytes(json.dumps(job_return_msg_dict).encode('UTF-8')))
    return 1


if __name__ == '__main__':
    fire.Fire()  # pragma: no cover
