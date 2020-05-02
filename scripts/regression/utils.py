import os
import glob
import json
from typing import Tuple

from spacer.messages import DataLocation
from spacer.data_classes import ImageLabels
from spacer.train_classifier import trainer_factory


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
                   n_epochs: int) -> None:
    feature_loc = DataLocation(storage_type='filesystem', key='')

    trainer = trainer_factory('minibatch')
    clf, val_results, return_message = trainer(
        train_labels, val_labels, n_epochs, [], feature_loc)
    with open(os.path.join(source_root, 'meta.json')) as fp:
        source_meta = json.load(fp)

    print('-> Re-trained {} ({}). Old acc: {:.1f}, new acc: {:.1f}'.format(
        source_meta['name'],
        source_meta['pk'],
        100 * float(source_meta['best_robot_accuracy']),
        100 * return_message.acc)
    )
