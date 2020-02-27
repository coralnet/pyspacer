import glob
import json
import os
import re

import boto
import fire
import tqdm

from spacer import config
from spacer.data_classes import ImageLabels
from spacer.messages import TrainClassifierMsg
from spacer.storage import storage_factory
from spacer.train_classifier import trainer_factory


class ClassifierRegressionTest:
    """
    This runs a training on exported features from CoralNet.
    It assumes read permission on the "spacer-trainingdata" bucket.

    All data is formatted per the management command in
    ...
    """

    # TODO: add permalink to management command --^

    @staticmethod
    def _cache_local(source_root, image_root, export_name, source_id):
        """ Download source data to local """
        conn = boto.connect_s3()
        bucket = conn.get_bucket('spacer-trainingdata', validate=True)
        if not os.path.exists(source_root):
            os.mkdir(source_root)
        if not os.path.exists(self.image_root):
            os.mkdir(image_root)

        mdkey = bucket.get_key('{}/s{}/meta.json'.format(
            export_name,
            source_id))
        mdkey.get_contents_to_filename(
            os.path.join(source_root, 'meta.json'))

        img_keys = bucket.list(prefix='{}/s{}/images'.format(
            export_name,
            source_id))

        img_keys = [key for key in img_keys if key.name.endswith('json')]

        print("-> Downloading {} metadata and feature files...".
              format(len(img_keys)))
        for key in tqdm.tqdm(img_keys):
            _, filename = key.name.split('images')
            local_path = os.path.join(image_root, filename.lstrip('/'))
            if not os.path.exists(local_path):
                key.get_contents_to_filename(local_path)

    def run(self,
            source_id: int,
            local_path: str,
            export_name: str = 'beta_export_v2'):

        source_root = os.path.join(local_path, 's{}'.format(source_id))
        image_root = os.path.join(source_root, 'images')

        # Download all data to local.
        # Train and eval Will run much faster that way.
        self._cache_local(source_root, image_root, export_name, source_id)

        # Create the train and val ImageLabels data structures.
        print("-> Assembling train and val data.")
        files = glob.glob(os.path.join(image_root, "*.json"))
        train_labels = ImageLabels(data={})
        val_labels = ImageLabels(data={})
        p = re.compile('([0-9]*).anns.json')
        for itt, filename in enumerate(files):
            if 'anns' in filename:
                with open(filename) as fp:
                    anns = json.load(fp)
                # Get primary key for this image.
                # This is how CoralNet divides up val vs train.
                # TODO: get train/val split from metadata file.
                pk = int(p.findall(filename)[0])
                if pk % 8 == 0:
                    labels = val_labels
                else:
                    labels = train_labels
                feature_key = filename.replace('anns', 'features')
                labels.data[feature_key] = [
                    (ann['row'], ann['col'], ann['label']) for ann in anns
                ]

        # Store and compile the TrainClassifierMsg
        storage = storage_factory('filesystem', '')
        traindata_key = os.path.join(source_root, 'traindata.json')
        valdata_key = os.path.join(source_root, 'valdata.json')
        storage.store_string(
            traindata_key,
            json.dumps(train_labels.serialize()))
        storage.store_string(
            valdata_key,
            json.dumps(val_labels.serialize()))

        msg = TrainClassifierMsg(
            pk=1,
            model_key='na',
            traindata_key=traindata_key,
            valdata_key=valdata_key,
            valresult_key='na',
            nbr_epochs=5,
            pc_models_key=[],
            pc_pks=[],
            bucketname='na',
            storage_type='filesystem'
        )

        # Perform training
        print("-> Training.")
        trainer = trainer_factory(msg, storage)
        clf, val_results, return_message = trainer()
        # TODO, compare to accuracy in source level metadata
        print(return_message.acc)

    @staticmethod
    def list(export_name: str = 'beta_export_v2'):
        """ Lists sources available in export. """

        conn = boto.connect_s3(config.AWS_ACCESS_KEY_ID,
                               config.AWS_SECRET_ACCESS_KEY)
        bucket = conn.get_bucket('spacer-trainingdata', validate=True)

        source_keys = bucket.list(prefix='{}/s'.format(export_name))

        for source_key in source_keys:

            print(source_key.name)
            meta_key_name = '{}/{}/meta.json'.format(export_name,
                                                     source_key.name)
            print(meta_key_name)
            meta_key = bucket.get_key(meta_key_name)

            md = json.loads(meta_key.get_content_to_string())
            print(md)


if __name__ == '__main__':
    fire.Fire(ClassifierRegressionTest)
