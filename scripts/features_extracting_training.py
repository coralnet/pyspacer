"""
This file is used for:
    1. run feature extractor
    2. train the classifier
on the exported spacer training data and confirms that
it get similar (or higher) performance than production.
"""

import os
import tqdm
import json
from io import BytesIO
from boto.s3.key import Key

from spacer import config
from spacer.messages import DataLocation, ExtractFeaturesMsg
from spacer.tasks import extract_features


class ExtractFeatures:
    """
    1. Initialize FeatureExtractor
    2. Load images and anns.meta from s3
    3. Extract features and save to local
    """
    def __init__(self, extractor_name):
        self.extractor_name = extractor_name

    def __call__(self, export_name, source_id):
        conn = config.get_s3_conn()
        bucket = conn.get_bucket('spacer-trainingdata', validate=True)

        # TODO: read source meta.json for performance comparison
        mdkey = bucket.get_key('{}/s{}/meta.json'.format(export_name, source_id))
        source_meta = json.load(BytesIO(mdkey.get_contents_as_string()))

        print("-> Fetching {} image and annotation meta files...".format(source_id))
        all_keys = bucket.list(prefix='{}/s{}/images'.format(export_name, source_id))
        img_keys = [key.name for key in all_keys if key.name.endswith('jpg')]
        ann_keys = [Key(bucket=bucket, name=img.replace('jpg', 'anns.json')) for img in img_keys]

        print("-> Extracting features...")
        for idx, im_key in enumerate(tqdm.tqdm(img_keys, position=0)):
            anns = json.load(BytesIO(ann_keys[idx].get_contents_as_string()))
            rowcols = [(ann['row'], ann['col']) for ann in anns]

            img_loc = DataLocation(storage_type='s3',
                                   key=im_key,
                                   bucket_name='spacer-trainingdata')
            feature_loc = DataLocation(
                storage_type='filesystem',
                key=os.path.join(
                    config.LOCAL_FEATURE_DIR,
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
    pass


if __name__ == '__main__':
    extractor = ExtractFeatures(extractor_name='efficientnet_b0_ver1')
    extractor(export_name='beta_export', source_id=294)
