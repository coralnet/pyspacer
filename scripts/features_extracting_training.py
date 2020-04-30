"""
This file is used for:
    1. run feature extractor
    2. train the classifier
on the exported spacer training data and confirms that
it get similar (or higher) performance than production.

To use do:
python scripts/features_extracting_training.py \
    efficientnet_b0_ver1 294 10 /path/to/features
"""

import os
import fire
import json
from io import BytesIO
from boto.s3.key import Key

from spacer import config
from spacer.tasks import extract_features
from spacer.messages import DataLocation, ExtractFeaturesMsg

from scripts.scripts_utils import build_traindata, start_training


class ExtractFeatures:
    """
    1. Initialize FeatureExtractor
    2. Load images and anns.meta from s3
    3. Extract features and save to local
    """
    def __init__(self, extractor_name, local_path):
        self.extractor_name = extractor_name
        self.local_path = local_path

    def __call__(self, source_id, export_name='beta_export'):
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
                rowcols = [(ann['row'], ann['col']) for ann in anns]

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
    def __init__(self, local_path):
        self.local_path = local_path

    @staticmethod
    def _cache(source_id, source_root, feats_root, export_name='beta_export'):
        conn = config.get_s3_conn()
        bucket = conn.get_bucket('spacer-trainingdata', validate=True)

        if not os.path.isdir(source_root):
            os.system('mkdir -p ' + source_root)
        if not os.path.isdir(feats_root):
            os.system('mkdir -p ' + feats_root)

        mdkey = bucket.get_key('{}/s{}/meta.json'.format(
            export_name, source_id
        ))
        all_keys = bucket.list(prefix='{}/s{}/images'.format(
            export_name, source_id
        ))
        anns_keys = [key for key in all_keys
                     if key.name.endswith('.anns.json')]
        meta_keys = [key for key in all_keys
                     if key.name.endswith('.meta.json')]
        assert len(anns_keys) == len(meta_keys)

        print("-> Downloading {} metadata...".format(len(anns_keys)))
        mdkey.get_contents_to_filename(os.path.join(source_root, 'meta.json'))
        for idx in range(len(anns_keys)):
            _, anns_filename = anns_keys[idx].name.split('images/')
            _, meta_filename = meta_keys[idx].name.split('images/')
            anns_local = os.path.join(feats_root, anns_filename)
            meta_local = os.path.join(feats_root, meta_filename)
            if not os.path.exists(anns_local):
                anns_keys[idx].get_contents_to_filename(anns_local)
            if not os.path.exists(meta_local):
                meta_keys[idx].get_contents_to_filename(meta_local)

    def __call__(self, source_id, n_epochs, export_name='beta_export'):

        print('-> Downloading data for source id: {}.'.format(source_id))
        source_root = os.path.join(self.local_path, 's{}'.format(source_id))
        feats_root = os.path.join(source_root, 'images')
        self._cache(
            source_id, source_root, feats_root, export_name
        )

        # Create the train and val ImageLabels data structures.
        print('-> Assembling train and val data for source id: {}'.format(
            source_id))
        train_labels, val_labels = build_traindata(feats_root)

        # Perform training
        print("-> Training...")
        start_training(source_root, train_labels, val_labels, n_epochs)


def main(extractor_name, source_id, n_epochs, local_path):
    extractor = ExtractFeatures(extractor_name, local_path)
    extractor(source_id)
    classifier = TrainClassifier(local_path)
    classifier(source_id, n_epochs)


if __name__ == '__main__':
    fire.Fire(main)
