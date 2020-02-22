import json
import re
import boto

from spacer.messages import \
    ImageLabels, \
    TrainClassifierMsg
from spacer.storage import storage_factory
from spacer.train_classifier import trainer_factory


class Source1070Regression:
    """
    This runs a training on exported features from CoralNet for source 1070.
    https://coralnet.ucsd.edu/source/1070/
    This source has 687 images.
    Training should result in similar accuracy that we saw on CoralNet (68%)

    All data is available in spacer-test/s1070/
    and it is formatted per the management command in
    https://github.com/beijbom/coralnet/blob/547fedcaea109d329a36fa8c66b17083d09594c6/project/vision_backend/management/commands/vb_export_spacer_data.py
    """

    @staticmethod
    def clean_up():
        storage = storage_factory('s3', 'spacer-test')
        storage.delete('tmp/valdata')
        storage.delete('tmp/traindata')

    @staticmethod
    def run():
        # Read out the Train and Val ImageLabels data structures.
        p = re.compile('([0-9]*).anns.json')
        conn = boto.connect_s3()
        bucket = conn.get_bucket('spacer-test')
        objects = bucket.list(prefix='s1070/images')
        train_labels = ImageLabels(data={})
        val_labels = ImageLabels(data={})
        for itt, key in enumerate(objects):
            if 'anns' in key.name:
                anns = json.loads(key.get_contents_as_string().decode('UTF-8'))
                # Get primary key for this image.
                # This is how CoralNet divides up val vs train.
                pk = int(p.findall(key.name)[0])
                if pk % 8 == 0:
                    labels = val_labels
                else:
                    labels = train_labels
                feature_key = key.name.replace('anns', 'features')
                labels.data[feature_key] = [
                    (ann['row'], ann['col'], ann['label']) for ann in anns
                ]

        # Store and compile the TrainClassifierMsg
        storage = storage_factory('s3', 'spacer-test')
        storage.store_string('tmp/traindata', json.dumps(train_labels.serialize()))
        storage.store_string('tmp/valdata', json.dumps(val_labels.serialize()))

        msg = TrainClassifierMsg(
            pk=1,
            model_key='dummy',
            traindata_key='tmp/traindata',
            valdata_key='tmp/valdata',
            valresult_key='dummy',
            nbr_epochs=5,
            pc_models_key=[],
            pc_pks=[],
            bucketname='spacer-test',
            storage_type='s3'
        )

        # Perform training
        trainer = trainer_factory(msg, storage)
        clf, val_results, return_message = trainer()

        print(return_message)
        print(val_results)


if __name__ == "__main__":
    reg = Source1070Regression()
    reg.run()
    reg.clean_up()
