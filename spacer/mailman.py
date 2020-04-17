"""
Defines the highest-level method for task handling through AWS SQS.
"""

import json

from spacer import config
from spacer.messages import JobMsg, JobReturnMsg
from spacer.tasks import \
    extract_features, \
    train_classifier, \
    classify_features, \
    classify_image


def sqs_mailman(in_queue='spacer_jobs',
                out_queue='spacer_results') -> bool:  # pragma: no cover
    """
    Looks for jobs in AWS SQS in_queue, process the job and writes
    results back to out_queue
    """
    print("-> Grabbing message.")

    # Load default queue
    conn = config.get_sqs_conn()
    in_queue = conn.get_queue(in_queue)
    out_queue = conn.get_queue(out_queue)

    # Read message
    m = in_queue.read()
    if m is None:
        print("-> No messages in inqueue.")
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
    m_out = out_queue.new_message(body=json.dumps(job_return_msg_dict))
    out_queue.write(m_out)
    in_queue.delete_message(m)
    return True


def process_job(job_msg: JobMsg) -> JobReturnMsg:

    run = {
        'extract_features': extract_features,
        'train_classifier': train_classifier,
        'classify_features': classify_features,
        'classify_image': classify_image,
    }

    assert isinstance(job_msg, JobMsg)
    assert job_msg.task_name in config.TASKS

    try:
        results = [run[job_msg.task_name](task) for task in job_msg.tasks]
        return_msg = JobReturnMsg(
            original_job=job_msg,
            ok=True,
            results=results,
            error_message=None
        )
    except Exception as e:
        return_msg = JobReturnMsg(
            original_job=job_msg,
            ok=False,
            results=None,
            error_message=repr(e)
        )
    return return_msg


if __name__ == '__main__':
    sqs_mailman()  # pragma: no cover
