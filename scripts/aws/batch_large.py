"""
This script submits multiple jobs and measures runtime across nbr points,
images size and extractor_name.
"""

from scripts.aws.utils import submit_jobs, monitor_jobs
import matplotlib.pyplot as plt
import numpy as np
import json


def run(job_queue: str = 'shakeout'):
    nbr_rowcols = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400,
                   500, 750, 1000]
    targets = []
    for im_size in [1000, 2500, 5000, 7500, 10000]:
        for name in ['efficientnet_b0_ver1', 'vgg16_coralnet_ver1']:
            targets.extend(submit_jobs(nbr_rowcols, job_queue, im_size, name))

    _, runtime = monitor_jobs(targets)

    to_store = []
    for job_id, _, job_msg, _, _, image_size in targets:
        to_store.append([
            job_id,
            len(job_msg.tasks[0].rowcols),
            job_msg.tasks[0].feature_extractor_name,
            image_size,
            runtime[job_id]
        ])

    json.dump(to_store, open('runtimes_shakeout.json', 'w'))


def render(filename='runtimes_shakeout.json'):

    data = json.load(open(filename))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    im_sizes = sorted(list(set([d[3] for d in data])))
    runtimes = [np.mean([d[4] for d in data if d[3] == im_size])
                for im_size in im_sizes]
    ax1.plot(im_sizes, runtimes, '-.')
    ax1.set_xlabel('Image size (nrows=ncols)')
    ax1.set_ylabel('Runtime (s)')

    nbr_rowcols = sorted(list(set([d[1] for d in data])))
    runtimes = [np.mean([d[4] for d in data if d[1] == nbr_rowcol])
                for nbr_rowcol in nbr_rowcols]
    ax2.plot(nbr_rowcols, runtimes, '-.')
    ax2.set_xlabel('Number of point locations')
    ax2.set_ylabel('Runtime (s)')

    names = sorted(list(set([d[2] for d in data])))
    runtimes = [np.mean([d[4] for d in data if d[2] == name])
                for name in names]
    ax3.bar(names, runtimes)
    ax3.set_xlabel('Feature extractor')
    ax3.set_ylabel('Runtime (s)')

    # Plot a second figure with the raw data.
    linestyles = {
        'vgg16_coralnet_ver1': '-',
        'efficientnet_b0_ver1': '-.',
    }
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    for name in names:
        for im_size in im_sizes:
            rt = [np.mean([d[4] for d in data if
                           d[1] == nbr_rowcol and
                           d[2] == name and
                           d[3] == im_size])
                  for nbr_rowcol in nbr_rowcols]
            ax1.plot(nbr_rowcols, rt,
                     linestyle=linestyles[name],
                     label='{}: {}'.format(name, im_size))
    ax1.set_xlabel('Number of point locations')
    ax1.set_ylabel('Runtime (s)')
    ax1.legend()
    plt.show()


if __name__ == '__main__':
    render()
