from spacer import config


def sqs_status(queue_name):
    """
    Returns number of pending and ongoing jobs in the queue.
    """
    conn = config.get_sqs_conn()
    queue = conn.get_queue(queue_name)
    attr = queue.get_attributes()
    return int(attr['ApproximateNumberOfMessages']), int(
        attr['ApproximateNumberOfMessagesNotVisible'])


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
