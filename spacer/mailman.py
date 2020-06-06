"""
Defines the highest-level method for task handling through AWS SQS.
"""

import json
import logging

import fire

from spacer import config
from spacer.messages import JobMsg
from spacer.tasks import \
    process_job


def sqs_fetch(in_queue: str = 'spacer_test_jobs',  # pragma: no cover
              out_queue: str = 'spacer_test_results') -> bool:
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
        return False
    job_msg_dict = json.loads(m.get_body())
    # Try to deserialize message
    try:
        job_msg = JobMsg.deserialize(job_msg_dict)
        job_return_msg = process_job(job_msg)
        job_return_msg_dict = job_return_msg.serialize()
    except Exception as e:
        # Handle deserialization errors directly in mailman.
        # All other errors are handled in "handle_message" function.
        job_return_msg_dict = {
            'original_job': job_msg_dict,
            'ok': False,
            'results': None,
            'error_message': 'Error deserializing message: ' + repr(e)
        }

    # Return
    logging.info("-> Writing results message to {}.".format(out_queue))
    m_out = out_queue.new_message(body=json.dumps(job_return_msg_dict))
    out_queue.write(m_out)
    in_queue.delete_message(m)
    return True


if __name__ == '__main__':
    fire.Fire()  # pragma: no cover
