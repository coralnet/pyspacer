"""
This file is used for:
    1. run feature extractor
    2. train the classifier
on the exported spacer training data and confirms that
it get similar (or higher) performance than production.
"""

import os
import json
import glob
from io import BytesIO
from boto.s3.key import Key

from spacer import config
from spacer.tasks import extract_features
from spacer.data_classes import ImageLabels
from spacer.train_classifier import trainer_factory
from spacer.messages import DataLocation, ExtractFeaturesMsg


class ExtractFeatures:
    """
    1. Initialize FeatureExtractor
    2. Load images and anns.meta from s3
    3. Extract features and save to local
    """
    def __init__(self, extractor_name):
        self.extractor_name = extractor_name

    def __call__(self, source_id, export_name='beta_export'):
        conn = config.get_s3_conn()
        bucket = conn.get_bucket('spacer-trainingdata', validate=True)

        print("-> Fetching {} image and annotation meta files...".format(source_id))
        all_keys = bucket.list(prefix='{}/s{}/images'.format(export_name, source_id))
        img_keys = [key.name for key in all_keys if key.name.endswith('jpg')]
        ann_keys = [Key(bucket=bucket, name=img.replace('jpg', 'anns.json')) for img in img_keys]

        feats_local_path = os.path.join(config.LOCAL_FEATURE_DIR, 's{}'.format(source_id), 'images')
        if not os.path.isdir(feats_local_path):
            os.system('mkdir -p ' + feats_local_path)

        print("-> Extracting features...")
        for idx, im_key in enumerate(img_keys):
            anns = json.load(BytesIO(ann_keys[idx].get_contents_as_string()))
            rowcols = [(ann['row'], ann['col']) for ann in anns]

            img_loc = DataLocation(storage_type='s3',
                                   key=im_key,
                                   bucket_name='spacer-trainingdata')
            feature_loc = DataLocation(
                storage_type='filesystem',
                key=os.path.join(
                    feats_local_path,
                    im_key.split('/')[-1].replace('jpg', 'features.json')
                )
            )

            msg = ExtractFeaturesMsg(
                job_token='extract_job',
                feature_extractor_name=self.extractor_name,
                rowcols=rowcols,
                image_loc=img_loc,
                feature_loc=feature_loc
            )
            return_msg = extract_features(msg)


class TrainClassifier:
    """
    1. Download meta.json and anns.json to local
    2. Train the classifier
    """
    @staticmethod
    def _cache(source_id, source_root, feats_root, export_name='beta_export'):
        conn = config.get_s3_conn()
        bucket = conn.get_bucket('spacer-trainingdata', validate=True)

        if not os.path.isdir(source_root):
            os.system('mkdir -p ' + source_root)
        if not os.path.isdir(feats_root):
            os.system('mkdir -p ' + feats_root)

        mdkey = bucket.get_key('{}/s{}/meta.json'.format(export_name, source_id))
        all_keys = bucket.list(prefix='{}/s{}/images'.format(export_name, source_id))
        anns_keys = [key for key in all_keys if key.name.endswith('.anns.json')]
        meta_keys = [key for key in all_keys if key.name.endswith('.meta.json')]
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
        source_root = os.path.join(config.LOCAL_FEATURE_DIR, 's{}'.format(source_id))
        feats_root = os.path.join(source_root, 'images')
        self._cache(source_id, source_root, feats_root, export_name='beta_export')

        # Create the train and val ImageLabels data structures.
        print('-> Assembling train and val data for source id: {}'.format(
            source_id))
        files = glob.glob(os.path.join(source_root, 'images', "*.json"))
        train_labels = ImageLabels(data={})
        val_labels = ImageLabels(data={})
        for idx, filename in enumerate(files):
            if 'anns' in filename:
                with open(filename, 'r') as fp:
                    anns = json.load(fp)
                meta_filename = filename.replace('anns', 'meta')
                with open(meta_filename, 'r') as fp:
                    meta = json.load(fp)
                if meta['in_trainset']:
                    labels = train_labels
                else:
                    assert meta['in_valset']
                    labels = val_labels
                labels.data[filename.replace('anns', 'features')] = [
                    (ann['row'], ann['col'], ann['label']) for ann in anns
                ]

        feature_loc = DataLocation(storage_type='filesystem', key='')
        # Perform training
        print("-> Training...")
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


if __name__ == '__main__':
    extractor = ExtractFeatures(extractor_name='efficientnet_b0_ver1')
    extractor(source_id=294)
    classifier = TrainClassifier()
    classifier(source_id=294, n_epochs=10)
