"""
This script submits 10 jobs to queues and monitors as the
jobs are completed.
"""
import logging
from typing import List

from scripts.aws.utils import submit_jobs, monitor_jobs


def main(nbr_rowcols: List[int],
         job_queue: str = 'shakeout',
         image_size: int = 100,
         extractor_name: str = 'efficientnet_b0_ver1'):

    logging.info("Starting scaling test for {}.".format(extractor_name))
    targets = submit_jobs(nbr_rowcols, job_queue, image_size, extractor_name)

    _, runtime = monitor_jobs(targets)

    for job_id, _, job_msg, _, _, _ in targets:
        print('{} pts: {:.2f} seconds'.format(len(job_msg.tasks[0].rowcols),
                                              runtime[job_id]))


if __name__ == '__main__':
    main(nbr_rowcols=[10, 20, 30, 40, 50, 60, 70, 80, 90])
