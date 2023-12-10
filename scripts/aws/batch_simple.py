"""
This script submits 10 jobs to queues and monitors as the
jobs are completed.
"""
from __future__ import annotations
import logging

from scripts.aws.utils import submit_jobs, monitor_jobs

logger = logging.getLogger(__name__)


def main(nbr_rowcols: list[int],
         job_queue: str = 'shakeout',
         image_size: int = 1000):

    targets = []
    for name in ['efficientnet_b0_ver1', 'vgg16_coralnet_ver1']:
        logger.info("Starting scaling test for {}.".format(name))
        targets.extend(submit_jobs(nbr_rowcols, job_queue, image_size, name))

    _, runtime = monitor_jobs(targets)

    for job_id, _, job_msg, _, _, _ in targets:
        print('{} pts using {}: {:.2f} seconds'.format(
            len(job_msg.tasks[0].rowcols),
            job_msg.tasks[0].feature_extractor_name,
            runtime[job_id]))


if __name__ == '__main__':
    main(nbr_rowcols=[10, 50, 100, 200])
