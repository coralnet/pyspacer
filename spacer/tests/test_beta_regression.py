"""
This file runs regression compared to results from the Beta production server.
See pyspacer/scripts/make_legacy_score_for_regression_testing.py for details.
"""
import json
import unittest

import numpy as np

from spacer import config
from spacer.storage import storage_factory
from spacer.messages import \
    DataLocation, \
    ExtractFeaturesMsg, \
    ClassifyFeaturesMsg, \
    ClassifyReturnMsg

from spacer.tasks import classify_features, extract_features

from spacer.data_classes import ImageFeatures

reg_meta = {
    's16': ('1355.model', ['i2921', 'i2934']),
    's295': ('10279.model', ['i1370227', 'i160100']),
    's603': ('3709.model', ['i576858', 'i576912']),
    's812': ('4772.model', ['i672762', 'i674185']),
    's1388': ('8942.model', ['i1023182', 'i1023213'])
}

s3_key_prefix = 'beta_reg/'


@unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
@unittest.skipUnless(config.HAS_S3_MODEL_ACCESS, 'No access to models')
@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestExtractFeatures(unittest.TestCase):
    def setUp(self):
        self.storage = storage_factory('s3', 'spacer-test')

        # Limit the number of row, col location to make tests run faster.
        self.max_rc_cnt = 2

    def get_rowcol(self, key):
        """ This file was saved using
        coralnet/project/vision_backend/management/commands/
        vb_export_spacer_data.py

        https://github.com/beijbom/coralnet/blob/
        e08afaa0164425fc16ae4ed60841d70f2eff59a6/project/vision_backend/
        management/commands/vb_export_spacer_data.py
        """
        print('Loading key: ' + key)
        anns = json.loads(self.storage.load(key).getvalue().decode('utf-8'))
        return [(entry['row'], entry['col']) for entry in anns]

    def run_test(self, key):
        """ Run feature extraction on an image and compare to legacy extracted
        features
        """
        new_feats_loc = DataLocation(storage_type='filesystem',
                                     key='/workspace/models/' +
                                         key.split('/')[1] +
                                         '.features.json')
        rowcol = self.get_rowcol(s3_key_prefix + key + '.anns.json')
        msg = ExtractFeaturesMsg(
            job_token='beta_reg_test',
            feature_extractor_name='vgg16_coralnet_ver1',
            image_loc=DataLocation(storage_type='s3',
                                   bucket_name='spacer-test',
                                   key=s3_key_prefix + key + '.jpg'),
            rowcols=rowcol[:self.max_rc_cnt],
            feature_loc=new_feats_loc
        )
        _ = extract_features(msg)

        legacy_feats = ImageFeatures.load(
            DataLocation(
                storage_type='s3',
                bucket_name='spacer-test',
                key=s3_key_prefix + key + '.features.json'
            ))

        self.assertFalse(legacy_feats.valid_rowcol)
        self.assertEqual(legacy_feats.npoints, len(rowcol))
        self.assertEqual(legacy_feats.feature_dim, 4096)

        new_feats = ImageFeatures.load(new_feats_loc)

        self.assertTrue(new_feats.valid_rowcol)
        self.assertEqual(new_feats.npoints, len(msg.rowcols))
        self.assertEqual(new_feats.feature_dim, 4096)

        for legacy_pf, new_pf in zip(legacy_feats.point_features,
                                     new_feats.point_features):
            print(key, np.linalg.norm(legacy_pf.data, new_pf.data))
            # self.assertTrue(np.allclose(legacy_pf.data, new_pf.data))

    def test_all(self):

        im_prefixes = []
        for source, (model, imgs) in reg_meta.items():
            for img in imgs:
                im_prefixes.append(source + '/' + img)

        for prefix in im_prefixes:
            self.run_test(prefix)


@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to tests')
class TestClassifyFeatures(unittest.TestCase):
    """ Test the classify_features task and compare to scores
    calculated using previous sci-kit learn versions.
    """

    def run_one_test(self, clf_key, img_key):

        msg = ClassifyFeaturesMsg(
            job_token='regression_test',
            feature_loc=DataLocation(storage_type='s3',
                                     bucket_name='spacer-test',
                                     key=s3_key_prefix + img_key +
                                         '.features.json'),
            classifier_loc=DataLocation(storage_type='s3',
                                        bucket_name='spacer-test',
                                        key=s3_key_prefix + clf_key)
        )
        new_scores = classify_features(msg)

        # The features are legacy, so the scores don't have valid row-cols.
        self.assertFalse(new_scores.valid_rowcol)

        legacy_scores = ClassifyReturnMsg.load(
            DataLocation(
                storage_type='s3',
                bucket_name='spacer-test',
                key=s3_key_prefix + img_key + '.scores.json'
            )
        )

        # The features are legacy, so the scores don't have valid row-cols.
        self.assertFalse(legacy_scores.valid_rowcol)
        for ls, ns in zip(legacy_scores.scores, new_scores.scores):
            self.assertTrue(np.allclose(ls[2], ns[2]))

    def test_all(self):

        for src_str in reg_meta:
            for img_str in reg_meta[src_str][1]:
                self.run_one_test(src_str + '/' + reg_meta[src_str][0],
                                  src_str + '/' + img_str)


if __name__ == '__main__':
    unittest.main()
