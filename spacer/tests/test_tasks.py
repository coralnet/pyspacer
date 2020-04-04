import json
import os
import unittest
import warnings

import numpy as np
from PIL import Image

from spacer import config
from spacer.data_classes import ImageFeatures
from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    ClassifyFeaturesMsg, \
    ClassifyImageMsg, \
    ClassifyReturnMsg, \
    DataLocation
from spacer.storage import store_classifier, load_classifier, storage_factory
from spacer.tasks import \
    extract_features, \
    train_classifier, \
    classify_features, \
    classify_image
from spacer.train_utils import make_random_data
from spacer.train_utils import train


class TestExtractFeatures(unittest.TestCase):

    def setUp(self):

        self.tmps = {
            'in': 'my_img.jpg',
            'out': 'my_output.json'
        }
        im = Image.new('RGB', (100, 100))
        im.save(self.tmps['in'])

    def tearDown(self):

        for tmp in self.tmps.values():
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_default(self):

        msg = ExtractFeaturesMsg(
            job_token='test',
            feature_extractor_name='dummy',
            image_loc=DataLocation(storage_type='filesystem',
                                   key=self.tmps['in']),
            rowcols=[(1, 1), (2, 2)],
            feature_loc=DataLocation(storage_type='filesystem',
                                     key=self.tmps['out'])
        )
        return_msg = extract_features(msg)
        self.assertTrue(type(return_msg) == ExtractFeaturesReturnMsg)
        self.assertTrue(os.path.exists(self.tmps['out']))


@unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
@unittest.skipUnless(config.HAS_S3_MODEL_ACCESS, 'No access to models')
@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestExtractFeaturesRegression(unittest.TestCase):

    def setUp(self):
        self.storage = storage_factory('s3', 'spacer-test')

    def get_rowcol(self, key):
        """ This file was saved using
        coralnet/project/vision_backend/management/commands/vb_export_spacer_data.py
        https://github.com/beijbom/coralnet/blob/e08afaa0164425fc16ae4ed60841d70f2eff59a6/project/vision_backend/management/commands/vb_export_spacer_data.py
        """
        anns = json.loads(self.storage.load(key).getvalue().decode('utf-8'))
        return [(entry['row'], entry['col']) for entry in anns]

    def run_test(self, prefix):
        """ Run feature extraction on an image and compare to legacy extracted
        features
        """

        msg = ExtractFeaturesMsg(
            job_token='regtest',
            feature_extractor_name='vgg16_coralnet_ver1',
            image_loc=DataLocation(storage_type='s3',
                                   bucket_name='spacer-test',
                                   key=prefix + '.jpg'),
            rowcols=self.get_rowcol(prefix + '.anns.json'),
            feature_loc=DataLocation(storage_type='memory',
                                     key=prefix + '.features.json')
        )
        _ = extract_features(msg)

        legacy_feats = ImageFeatures.load(
            DataLocation(
                storage_type='s3',
                bucket_name='spacer-test',
                key=prefix + '.features.json'
            ))

        self.assertFalse(legacy_feats.valid_rowcol)
        self.assertEqual(legacy_feats.npoints, len(msg.rowcols))
        self.assertEqual(legacy_feats.feature_dim, 4096)

        new_feats = ImageFeatures.load(
            DataLocation(
                storage_type='memory',
                key=prefix + '.features.json'
            ))

        self.assertTrue(new_feats.valid_rowcol)
        self.assertEqual(new_feats.npoints, len(msg.rowcols))
        self.assertEqual(new_feats.feature_dim, 4096)

        for legacy_pf, new_pf in zip(legacy_feats.point_features,
                                     new_feats.point_features):
            self.assertTrue(np.allclose(legacy_pf.data, new_pf.data))

    def test_legacy(self):

        prefixes = ['s16/i2934',
                    's16/i2921',
                    's295/i1370227',
                    's295/i160100']

        for prefix in prefixes:
            self.run_test(prefix)


class TestTrainClassifier(unittest.TestCase):

    def test_default(self):

        # Set some hyper parameters for data generation
        n_valdata = 20
        n_traindata = 200
        points_per_image = 20
        feature_dim = 5
        class_list = [1, 2]

        # Create train and val data.
        features_loc_template = DataLocation(storage_type='memory', key='')

        traindata_loc = DataLocation(storage_type='memory', key='traindata')
        traindata = make_random_data(n_valdata,
                                     class_list,
                                     points_per_image,
                                     feature_dim,
                                     features_loc_template)
        traindata.store(traindata_loc)

        valdata = make_random_data(n_traindata,
                                   class_list,
                                   points_per_image,
                                   feature_dim,
                                   features_loc_template)
        valdata_loc = DataLocation(storage_type='memory', key='traindata')
        valdata.store(valdata_loc)

        # Train once by calling directly so that we have a previous classifier.
        clf, _ = train(traindata, features_loc_template, 1)

        previous_classifier_loc = DataLocation(storage_type='memory',
                                               key='pc')
        store_classifier(previous_classifier_loc, clf)

        msg = TrainClassifierMsg(
            job_token='test',
            trainer_name='minibatch',
            nbr_epochs=1,
            traindata_loc=traindata_loc,
            valdata_loc=valdata_loc,
            features_loc=features_loc_template,
            previous_model_locs=[previous_classifier_loc],
            model_loc=DataLocation(storage_type='memory', key='model'),
            valresult_loc=DataLocation(storage_type='memory', key='val_res')
        )
        return_msg = train_classifier(msg)
        self.assertTrue(type(return_msg) == TrainClassifierReturnMsg)


class TestClassifyFeatures(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)

    @unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to tests')
    def test_legacy(self):
        msg = ClassifyFeaturesMsg(
            job_token='my_job',
            feature_loc=DataLocation(storage_type='s3',
                                     bucket_name='spacer-test',
                                     key='legacy.jpg.feats'),
            classifier_loc=DataLocation(storage_type='s3',
                                        key='legacy.model',
                                        bucket_name='spacer-test')
        )

        return_msg = classify_features(msg)

        # Legacy feature didn't store rowcol information.
        self.assertFalse(return_msg.valid_rowcol)

        self.assertRaises(ValueError, return_msg.__getitem__, (10, 20))

        for row, col, scores in return_msg.scores:
            self.assertEqual(row, None)
            self.assertEqual(col, None)
            self.assertTrue(isinstance(scores, list))
        self.assertTrue(type(return_msg.scores), ClassifyReturnMsg)

    @unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to tests')
    def test_new(self):

        feats = ImageFeatures.make_random([1, 2, 3, 2], feature_dim=4096)
        feature_loc = DataLocation(storage_type='memory',
                                   key='new.jpg.feats')
        feats.store(feature_loc)

        model_loc = DataLocation(storage_type='s3',
                                 key='legacy.model',
                                 bucket_name='spacer-test')

        msg = ClassifyFeaturesMsg(
            job_token='my_job',
            feature_loc=feature_loc,
            classifier_loc=model_loc
        )

        clf = load_classifier(model_loc)

        return_msg = classify_features(msg)

        # Legacy feature didn't store rowcol information.
        self.assertTrue(return_msg.valid_rowcol)
        for pf in feats.point_features:
            self.assertTrue(isinstance(return_msg[(pf.row, pf.col)], list))

        for row, col, scores in return_msg.scores:
            self.assertTrue(isinstance(row, int))
            self.assertTrue(isinstance(col, int))
            self.assertTrue(isinstance(scores, list))
            self.assertEqual(len(scores), len(clf.classes_))

        self.assertTrue(type(return_msg.scores), ClassifyReturnMsg)


@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to tests')
class TestClassifyFeaturesLegacy(unittest.TestCase):
    """ Test the classify_features task and compare to scores
    calculated using previous sci-kit learn versions.
    """

    def run_one_test(self, clf_str, img_prefix):

        msg = ClassifyFeaturesMsg(
            job_token='regression_test',
            feature_loc=DataLocation(storage_type='s3',
                                     bucket_name='spacer-test',
                                     key=img_prefix + '.features.json'),
            classifier_loc=DataLocation(storage_type='s3',
                                        bucket_name='spacer-test',
                                        key=clf_str)
        )
        new_scores = classify_features(msg)

        # The features are legacy, so the scores don't have valid row-cols.
        self.assertFalse(new_scores.valid_rowcol)

        legacy_scores = ClassifyReturnMsg.load(
            DataLocation(
                storage_type='s3',
                bucket_name='spacer-test',
                key=img_prefix + '.scores.json'
            )
        )

        # The features are legacy, so the scores don't have valid row-cols.
        self.assertFalse(legacy_scores.valid_rowcol)
        for ls, ns in zip(legacy_scores.scores, new_scores.scores):
            self.assertTrue(np.allclose(ls[2], ns[2]))

    def test_all(self):

        meta = {
            's16': ('1355.model', ['i2921', 'i2934']),
            's295': ('10279.model', ['i1370227', 'i160100'])
        }

        for src_str in meta:
            for img_str in meta[src_str][1]:
                self.run_one_test(src_str + '/' + meta[src_str][0],
                                  src_str + '/' + img_str)


class TestClassifyImage(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)

    @unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to tests')
    def test_deploy_simple(self):
        msg = ClassifyImageMsg(
            job_token='my_job',
            image_loc=DataLocation(storage_type='url',
                                   key='https://homepages.cae.wisc.edu/~ece533'
                                       '/images/baboon.png'),
            feature_extractor_name='dummy',
            rowcols=[(100, 100), (200, 200)],
            classifier_loc=DataLocation(storage_type='s3',
                                        key='legacy.model',
                                        bucket_name='spacer-test')
        )

        return_msg = classify_image(msg)
        self.assertEqual(len(return_msg.scores), len(msg.rowcols))
        for rowcol in msg.rowcols:
            self.assertTrue(isinstance(return_msg[rowcol], list))

        for row, col, scores in return_msg.scores:
            self.assertTrue(isinstance(row, int))
            self.assertTrue(isinstance(col, int))
            self.assertTrue(isinstance(scores, list))

        for rowcol, rc_score in zip(msg.rowcols, return_msg.scores):
            self.assertEqual(rowcol, rc_score[:2])

        self.assertTrue(type(return_msg.scores), ClassifyReturnMsg)


class TestClassifyImageCache(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)

    @unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to tests')
    def test_classify_image_with_caching(self):
        """ Call classify_image three times.
        The first 2 time with same message.
        The last time with a new message (different classifier).
        Due to caching, the second call should be the fastest of the three.
        """

        load_classifier.cache_clear()
        msg = ClassifyImageMsg(
            job_token='my_job',
            image_loc=DataLocation(storage_type='url',
                                   key='https://homepages.cae.wisc.edu/~ece533'
                                       '/images/baboon.png'),
            feature_extractor_name='dummy',
            rowcols=[(100, 100), (200, 200)],
            classifier_loc=DataLocation(storage_type='s3',
                                        key='legacy.model',
                                        bucket_name='spacer-test')
        )

        msg2 = ClassifyImageMsg(
            job_token='my_job',
            image_loc=DataLocation(storage_type='url',
                                   key='https://homepages.cae.wisc.edu/~ece533'
                                       '/images/baboon.png'),
            feature_extractor_name='dummy',
            rowcols=[(100, 100), (200, 200)],
            classifier_loc=DataLocation(storage_type='s3',
                                        key='legacy_model2.pkl',
                                        bucket_name='spacer-test')
        )

        return_msg1 = classify_image(msg)
        return_msg2 = classify_image(msg)
        return_msg3 = classify_image(msg2)
        self.assertLess(return_msg2.runtime, return_msg1.runtime)
        self.assertLess(return_msg2.runtime, return_msg3.runtime)


class TestBadRowcols(unittest.TestCase):

    def test_image_classify(self):
        msg = ClassifyImageMsg(
            job_token='my_job',
            image_loc=DataLocation(storage_type='url',
                                   key='https://homepages.cae.wisc.edu/~ece533'
                                       '/images/baboon.png'),
            feature_extractor_name='dummy',
            rowcols=[(-1, -1)],
            classifier_loc=DataLocation(storage_type='s3',
                                        key='legacy.model',
                                        bucket_name='spacer-test')
        )

        try:
            classify_image(msg)
            raise ValueError("classify_image should raise an error.")
        except AssertionError as err:
            self.assertIn('negative', repr(err))
            self.assertIn('-1', repr(err))


if __name__ == '__main__':
    unittest.main()
