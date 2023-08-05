"""
This file provides scripts for
1) listing all extracted data in spacer-trainingdata bucket.
2) Re-train a classifier and compared to performance on CoralNet
"""


import json
import os

import fire
import logging
import warnings

from spacer import config
from scripts.regression.utils import build_traindata, do_training, \
    cache_local, do_extract_features


def train(source_id: int,
          local_path: str,
          n_epochs: int = 5,
          export_name: str = 'beta_export',
          clf_type: str = 'MLP',
          extractor_name: str = None) -> None:
    """
    This runs a training on exported features from CoralNet.
    It assumes read permission on the "spacer-trainingdata" bucket.

    All data is formatted per the management command in
    https://github.com/coralnet/coralnet/blob/107257fd34cd2c16714b369ec7146ae7222af2c6/project/vision_backend/management/commands/vb_export_spacer_data.py
    ...
    """

    # Sk-learn calibration step throws out a ton of warnings.
    # That we don't need to see here.
    warnings.simplefilter('ignore', RuntimeWarning)
    config.filter_warnings()

    source_root = os.path.join(local_path, 's{}'.format(source_id))
    image_root = os.path.join(source_root, 'images')
    if not os.path.exists(image_root):
        os.makedirs(image_root)

    logging.basicConfig(format='%(asctime)s %(message)s',
                        filename=os.path.join(source_root, 'retrain.log'),
                        level=logging.INFO)

    # Download all data to local.
    if extractor_name is None:
        # Use the legacy pre-extracted features
        cache_local(source_root, image_root, export_name, source_id,
                    cache_image=False, cache_feats=True)
    else:
        # Re-extract features so download the images instead.
        cache_local(source_root, image_root, export_name, source_id,
                    cache_image=True, cache_feats=False)
        do_extract_features(extractor_name, image_root)

    # Build traindata
    train_labels, val_labels = build_traindata(image_root)

    # Perform training
    do_training(source_root, train_labels, val_labels, n_epochs, clf_type)


def list_sources(export_name: str = 'beta_export') -> None:
    """ Lists sources available in export. """

    conn = config.get_s3_conn()
    bucket = conn.Bucket('spacer-trainingdata')

    obj_list = bucket.objects.filter(Prefix='{}/s'.format(export_name),
                                     Delimiter='images')
    obj_list = [obj for obj in obj_list if obj.key.endswith('json')]
    obj_list.sort(key=lambda obj: int(obj.key.split('/')[1][1:]))

    header_format = '{:>30}, {:>4}, {:>6}, {}\n{}'
    print(header_format.format('Name', 'id', 'n_imgs', 'acc (%)', '-'*53))
    entry_format = '{:>30}, {:>4}, {:>6}, {:.1f}%'

    for obj in obj_list:
        md = json.loads(obj.get()['Body'].read().decode('utf-8'))

        if not'pk' in md:
            # One source "Mestrado" was deleted before we could
            # refresh the export metadata. So get pk from the path.
            print(entry_format.format(
                md['name'][:20],
                obj.key.split('/')[1][1:],
                md['nbr_confirmed_images'], 0) + ' Old metadata!!')
        else:
            print(entry_format.format(
                md['name'][:20],
                md['pk'],
                md['nbr_confirmed_images'],
                100*float(md['best_robot_accuracy'])))


if __name__ == '__main__':
    fire.Fire()
