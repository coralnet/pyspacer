"""
This script submits multiple jobs and measures runtime across nbr points,
images size and extractor_name.
"""

from scripts.aws.utils import submit_jobs, monitor_jobs


def main(job_queue: str = 'shakeout'):
    nbr_rowcols = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500,
                   600, 700, 800, 900, 1000]
    targets = []
    for imsize in [1000, 3000, 5000, 7000, 10000]:
        for name in ['efficientnet_b0_ver1', 'vgg16_coralnet_ver1']:
            targets.extend(submit_jobs(nbr_rowcols, job_queue, imsize, name))

    _, runtime = monitor_jobs(targets)

    with open('output.txt', 'a') as f:
        for job_id, _, job_msg, _, _, image_size in targets:
            f.write('{}, {}, {}, {}, {}\n'.format(
                job_id,
                len(job_msg.tasks[0].rowcols),
                job_msg.tasks[0].feature_extractor_name,
                image_size,
                runtime[job_id]
            ))


if __name__ == '__main__':
    main()
