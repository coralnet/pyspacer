"""
This file is used for:
    1. run feature extractor
    2. train the classifier
on the exported spacer training data and confirms that
it get similar (or higher) performance than production.

To use do:
python scripts/regression/efficientnet_extractor.py \
    efficientnet_b0_ver1 294 10 MLP /path/to/features
"""

import os
import fire
import json
from io import BytesIO
from boto.s3.key import Key

from spacer import config
from spacer.tasks import extract_features
from spacer.messages import DataLocation, ExtractFeaturesMsg

from scripts.regression.utils import build_traindata, \
    start_training, \
    cache_local


class ExtractFeatures:
    """
    1. Initialize FeatureExtractor
    2. Load images and anns.meta from s3
    3. Extract features and save to local
    """
    def __init__(self,
                 extractor_name: str,
                 local_path: str) -> None:
        self.extractor_name = extractor_name
        self.local_path = local_path

    def __call__(self,
                 source_id: int,
                 export_name: str = 'beta_export') -> None:
        conn = config.get_s3_conn()
        bucket = conn.get_bucket('spacer-trainingdata', validate=True)

        print("-> Fetching {} image and annotation meta files...".format(
            source_id))
        all_keys = bucket.list(prefix='{}/s{}/images'.format(
            export_name, source_id))
        img_keys = [key.name for key in all_keys if key.name.endswith('jpg')]
        ann_keys = [Key(bucket=bucket, name=img.replace('jpg', 'anns.json'))
                    for img in img_keys]

        feats_local_path = os.path.join(self.local_path, 's{}'.format(
            source_id), 'images')
        if not os.path.isdir(feats_local_path):
            os.system('mkdir -p ' + feats_local_path)

        print("-> Extracting features...")
        for idx, im_key in enumerate(img_keys):
            feature_path = os.path.join(
                feats_local_path,
                im_key.split('/')[-1].replace('jpg', 'features.json')
            )
            if not os.path.exists(feature_path):
                anns = json.load(BytesIO(
                    ann_keys[idx].get_contents_as_string()
                ))
                rowcols = [(ann['row']-1, ann['col']-1) for ann in anns]

                img_loc = DataLocation(storage_type='s3',
                                       key=im_key,
                                       bucket_name='spacer-trainingdata')
                feature_loc = DataLocation(
                    storage_type='filesystem',
                    key=feature_path
                )
                msg = ExtractFeaturesMsg(
                    job_token='extract_job',
                    feature_extractor_name=self.extractor_name,
                    rowcols=rowcols,
                    image_loc=img_loc,
                    feature_loc=feature_loc
                )
                _ = extract_features(msg)


class TrainClassifier:
    """
    1. Download meta.json and anns.json to local
    2. Train the classifier
    """
    def __init__(self,
                 local_path: str) -> None:
        self.local_path = local_path

    def __call__(self,
                 source_id: int,
                 n_epochs: int = 5,
                 export_name: str = 'beta_export',
                 clf_type: str = 'MLP') -> None:

        print('-> Downloading data for source id: {}.'.format(source_id))
        source_root = os.path.join(self.local_path, 's{}'.format(source_id))
        image_root = os.path.join(source_root, 'images')

        # Download all data to local.
        # Train and eval will run much faster that way...
        cache_local(source_root, image_root, export_name, source_id)

        # Create the train and val ImageLabels data structures.
        print('-> Assembling train and val data for source id: {}'.format(
            source_id))
        train_labels, val_labels = build_traindata(image_root)

        # Perform training
        print("-> Training...")
        start_training(source_root, train_labels, val_labels, n_epochs,
                       clf_type)


def main(extractor_name: str,
         source_id: int,
         n_epochs: int,
         clf_type: str,
         local_path: str) -> None:
    extractor = ExtractFeatures(extractor_name, local_path)
    extractor(source_id)
    classifier = TrainClassifier(local_path)
    classifier(source_id, n_epochs, clf_type=clf_type)


if __name__ == '__main__':
    fire.Fire(main)
