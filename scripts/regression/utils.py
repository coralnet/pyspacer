import glob
import json
import tqdm
import os
from typing import Tuple

from spacer import config
from spacer.data_classes import ImageLabels
from spacer.messages import DataLocation
from spacer.train_classifier import trainer_factory


def cache_local(source_root: str,
                image_root: str,
                export_name: str,
                source_id: int) -> None:
    # Download data to the local to speed up training
    conn = config.get_s3_conn()
    bucket = conn.get_bucket('spacer-trainingdata', validate=True)
    if not os.path.exists(source_root):
        os.system('mkdir -p ' + source_root)
    if not os.path.exists(image_root):
        os.system('mkdir -p ' + image_root)

    mdkey = bucket.get_key('{}/s{}/meta.json'.format(export_name, source_id))
    mdkey.get_contents_to_filename(os.path.join(source_root, 'meta.json'))
    img_keys = bucket.list(prefix='{}/s{}/images'.format(export_name,
                                                         source_id))
    img_keys = [key for key in img_keys if key.name.endswith('json')]

    print("-> Downloading {} metadata and feature files...".
          format(len(img_keys)))
    for key in tqdm.tqdm(img_keys):
        _, filename = key.name.split('images')
        local_path = os.path.join(image_root, filename.lstrip('/'))
        if not os.path.exists(local_path):
            key.get_contents_to_filename(local_path)


def build_traindata(image_root: str) -> Tuple[ImageLabels, ImageLabels]:
    # Create the train and val ImageLabels data structures.
    ann_files = glob.glob(os.path.join(image_root, "*.anns.json"))
    train_labels = ImageLabels(data={})
    val_labels = ImageLabels(data={})
    for itt, ann_file in enumerate(ann_files):

        meta_file = ann_file.replace('anns', 'meta')
        features_file = ann_file.replace('anns', 'features')

        with open(ann_file) as fp:
            anns = json.load(fp)

        with open(meta_file) as fp:
            meta = json.load(fp)

        if meta['in_trainset']:
            labels = train_labels
        else:
            assert meta['in_valset']
            labels = val_labels

        labels.data[features_file] = [
            (ann['row']-1, ann['col']-1, ann['label']) for ann in anns
        ]
    return train_labels, val_labels


def start_training(source_root: str,
                   train_labels: ImageLabels,
                   val_labels: ImageLabels,
                   n_epochs: int,
                   clf_type: str) -> None:
    feature_loc = DataLocation(storage_type='filesystem', key='')

    trainer = trainer_factory('minibatch')
    clf, val_results, return_message = trainer(
        train_labels, val_labels, n_epochs, [], feature_loc, clf_type)
    with open(os.path.join(source_root, 'meta.json')) as fp:
        source_meta = json.load(fp)

    print('-> Re-trained {} ({}). Old acc: {:.1f}, new acc: {:.1f}'.format(
        source_meta['name'],
        source_meta['pk'],
        100 * float(source_meta['best_robot_accuracy']),
        100 * return_message.acc)
    )
