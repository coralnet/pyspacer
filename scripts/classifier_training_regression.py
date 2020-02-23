import json
import re
import boto
import os
import glob
import tqdm

from spacer.messages import \
    ImageLabels, \
    TrainClassifierMsg
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

    def __init__(self,
                 source_id: int,
                 local_path: str,
                 export_name: str='beta_export'):
        self.source_id = source_id
        self.local_path = local_path
        self.export_name = export_name
        self.source_root = os.path.join(self.local_path,
                                        's{}'.format(self.source_id))
        self.image_root = os.path.join(self.source_root, 'images')

    def _cache_local(self):
        """ Download source data to local """
        conn = boto.connect_s3()
        bucket = conn.get_bucket('spacer-trainingdata', validate=True)
        if not os.path.exists(self.source_root):
            os.mkdir(self.source_root)
        if not os.path.exists(self.image_root):
            os.mkdir(self.image_root)

        mdkey = bucket.get_key('{}/s{}/meta.json'.format(
            self.export_name,
            self.source_id))
        mdkey.get_contents_to_filename(
            os.path.join(self.source_root, 'meta.json'))

        img_keys = bucket.list(prefix='{}/s{}/images'.format(
            self.export_name,
            self.source_id))

        img_keys = [key for key in img_keys if key.name.endswith('json')]

        print("-> Downloading {} metadata and feature files...".
              format(len(img_keys)))
        for key in tqdm.tqdm(img_keys):
            _, filename = key.name.split('images')
            local_path = os.path.join(self.image_root, filename.lstrip('/'))
            if not os.path.exists(local_path):
                key.get_contents_to_filename(local_path)

    def __call__(self):

        # Download all data to local.
        # Train and eval Will run much faster that way.
        self._cache_local()

        # Create the train and val ImageLabels data structures.
        print("-> Assembling train and val data.")
        files = glob.glob(os.path.join(self.image_root, "*.json"))
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
        traindata_key = os.path.join(self.source_root, 'traindata.json')
        valdata_key = os.path.join(self.source_root, 'valdata.json')
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


if __name__ == "__main__":
    # TODO, add fire command line interface.
    reg = ClassifierRegressionTest(1070, '/Users/beijbom/tmp')
    reg()
