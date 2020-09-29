"""
Defines the highest-level method for task handling through AWS SQS.
"""

import json
import logging

import fire
import os

from spacer import config
from spacer.messages import JobMsg, JobReturnMsg, DataLocation
from spacer.tasks import \
    process_job


def run_job_verbose(job_msg_loc) -> JobReturnMsg:
    """ Deserializes and executes job message. """

    logging.info("-> Deserializing job message location...")
    job_msg_loc_dict = json.loads(job_msg_loc)
    job_location = DataLocation.deserialize(job_msg_loc_dict)
    logging.info("-> Done deserializing job message location.")
    logging.info("-> Instantiating job message...")
    job_msg = JobMsg.load(job_location)
    logging.info("-> Done instantiating job message: {}.")
    job_return_msg = process_job(job_msg)
    logging.info("-> Done processing job.")
    return job_return_msg


def env_job():
    """ Runs a job defined in environmental variables. Setup to play
    nicely with AWS Batch.
    Expects JOB_MSG_LOC to contain the json-serialized DataLocation of JobMgs.
    Expects OUT_QUEUE to contain the name of the desired results queue.
    """

    out_queue_name = os.getenv('OUT_QUEUE')
    if out_queue_name is None:
        out_queue_name = 'spacer_shakeout_results'

    job_msg_loc = os.getenv('JOB_MSG_LOC')
    logging.info("-> Received boto job for ENV {}.".format(job_msg_loc))

    try:
        job_return_msg_dict = run_job_verbose(job_msg_loc).serialize()

    except Exception as e:
        # Handle deserialization errors directly in mailman.
        # All other errors are handled in "handle_message" function.
        job_return_msg_dict = {
            'original_job': job_msg_loc,
            'ok': False,
            'results': None,
            'error_message': 'Error deserializing message: ' + repr(e)
        }

    # Return
    logging.info("-> Writing results to {}.".format(out_queue_name))
    conn = config.get_sqs_conn()
    out_queue = conn.get_queue(out_queue_name)
    m_out = out_queue.new_message(body=json.dumps(job_return_msg_dict))
    out_queue.write(m_out)
    logging.info("-> Done writing results to {}.".format(out_queue_name))
    return True


def sqs_fetch(in_queue: str = 'spacer_test_jobs',  # pragma: no cover
              out_queue: str = 'spacer_shakeout_results') -> bool:
    """
    Looks for jobs in AWS SQS in_queue, process the job and writes
    results back to out_queue
    :param in_queue: Name of AWS SQS from which to fetch the job.
    :param out_queue: Name of AWS SQS to which to store the results.
    """
    logging.info("-> Grabbing message from {}.".format(in_queue))

    # Load default queue
    conn = config.get_sqs_conn()
    in_queue = conn.get_queue(in_queue)
    out_queue = conn.get_queue(out_queue)

    # Read message
    m = in_queue.read()
    if m is None:
        logging.info("-> No messages in inqueue.")
        return True
    job_msg_loc = m.get_body()

    # Try to deserialize message
    try:
        job_return_msg_dict = run_job_verbose(job_msg_loc).serialize()
    except Exception as e:
        # Handle deserialization errors directly in mailman.
        # All other errors are handled in "handle_message" function.
        job_return_msg_dict = {
            'original_job': job_msg_loc,
            'ok': False,
            'results': None,
            'error_message': 'Error deserializing message: ' + repr(e)
        }

    # Return
    logging.info("-> Writing results to {}.".format(out_queue))
    m_out = out_queue.new_message(body=json.dumps(job_return_msg_dict))
    out_queue.write(m_out)
    in_queue.delete_message(m)
    return True


if __name__ == '__main__':
    fire.Fire()  # pragma: no cover
