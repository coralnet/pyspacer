import json
import unittest

import numpy as np

from spacer import config
from spacer.messages import \
    ImageLabels, \
    ImageFeatures, \
    PointFeatures, \
    TrainClassifierMsg
from spacer.storage import storage_factory
from spacer.train_classifier import trainer_factory


@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestDefaultTrainerDummyData(unittest.TestCase):

    def setUp(self):

        self.n_valdata = 20
        self.n_traindata = 200
        self.n_points = 10
        self.feature_dim = 5
        self.storage = storage_factory('memory', '')

    def make_image_features(self):

        pfs = [PointFeatures(row=itt,
                             col=itt,
                             data=list(np.random.multivariate_normal(
                                 np.ones(self.feature_dim)*itt % 2,
                                 np.eye(self.feature_dim))))
               for itt in range(self.n_points)]

        return ImageFeatures(point_features=pfs,
                             valid_rowcol=True,
                             feature_dim=self.feature_dim,
                             npoints=self.n_points)

    def make_labels(self, count):

        labels = ImageLabels(data={})
        for i in range(count):
            feats = self.make_image_features()
            imkey = 'val_{}'.format(i)
            self.storage.store_string(imkey, json.dumps(feats.serialize()))
            labels.data[imkey] = [
                (pf.row, pf.col, itt % 2) for itt, pf in
                enumerate(feats.point_features)
            ]
        return labels

    def test_gaussian(self):

        np.random.seed(0)
        labels = self.make_labels(self.n_valdata)
        self.storage.store_string('valdata', json.dumps(labels.serialize()))

        labels = self.make_labels(self.n_traindata)
        self.storage.store_string('traindata', json.dumps(labels.serialize()))

        msg = TrainClassifierMsg(
            pk=1,
            model_key='dummy',
            traindata_key='traindata',
            valdata_key='valdata',
            valresult_key='dummy',
            nbr_epochs=5,
            pc_models_key=[],
            pc_pks=[],
            bucketname='',
            storage_type='filesystem'
        )

        trainer = trainer_factory(msg, self.storage)
        clf, val_results, return_message = trainer()

        # The way we rendered the data, accuracy is usually around 90%.
        # Adding some margin to account for randomness.
        # TODO: fix random seed; somehow the set above didn't work.
        self.assertGreater(return_message.acc,
                           0.80,
                           "Failure may be due to random generated numbers,"
                           "re-run tests.")


if __name__ == '__main__':
    unittest.main()
