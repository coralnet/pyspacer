"""
These tests check for compatibility with results of previous coralnet/pyspacer
versions.
Much of the test data was prepared using scripts/regression/caffe_extractor.py
"""
import unittest

import numpy as np

from spacer import config
from spacer.data_classes import ImageFeatures
from spacer.messages import \
    DataLocation, \
    ExtractFeaturesMsg, \
    ClassifyFeaturesMsg, \
    ClassifyReturnMsg
from spacer.storage import storage_factory
from spacer.tasks import classify_features, extract_features
from spacer.tests.utils import cn_beta_fixture_location


cn_beta_fixtures = {
    's16': ('1355.model', ['i2921', 'i2934']),
    's295': ('10279.model', ['i1370227', 'i160100']),
    's603': ('3709.model', ['i576858', 'i576912']),
    's812': ('4772.model', ['i672762', 'i674185']),
    's1388': ('8942.model', ['i1023182', 'i1023213'])
}
pyspacer_031_vgg16_fixtures = {
    '23': {'classifier': 33358, 'images': [3286565, 3286615]},
    '1288': {'classifier': 33360, 'images': [3286481, 3286511]},
}
pyspacer_031_efficientnet_fixtures = {
    '23': {'classifier': 33369, 'images': [3286565, 3286615]},
    '1288': {'classifier': 33368, 'images': [3286481, 3286511]},
}


def pyspacer031_vgg16_fixture_location(key):
    return DataLocation(
        storage_type='s3',
        bucket_name=config.TEST_BUCKET,
        key='legacy_compat/pyspacer_0.3.1/vgg16/' + key
    )


def pyspacer031_efficientnet_fixture_location(key):
    return DataLocation(
        storage_type='s3',
        bucket_name=config.TEST_BUCKET,
        key='legacy_compat/pyspacer_0.3.1/efficientnet/' + key
    )


def extract_and_classify(im_key, clf_key, rowcol):
    """ Helper method for extract_and_classify regression tests. """
    new_feats_loc = DataLocation(storage_type='memory',
                                 key='features.json')

    msg = ExtractFeaturesMsg(
        job_token='beta_reg_test',
        feature_extractor_name='vgg16_coralnet_ver1',
        image_loc=cn_beta_fixture_location(im_key + '.jpg'),
        rowcols=rowcol,
        feature_loc=new_feats_loc
    )
    _ = extract_features(msg)

    msg = ClassifyFeaturesMsg(
        job_token='regression_test',
        feature_loc=new_feats_loc,
        classifier_loc=cn_beta_fixture_location(clf_key)
    )
    new_return = classify_features(msg)

    legacy_return = ClassifyReturnMsg.load(
        cn_beta_fixture_location(im_key + '.scores.json')
    )
    return new_return, legacy_return


@unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
@unittest.skipUnless(config.HAS_S3_MODEL_ACCESS, 'No access to models')
@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestExtractFeatures(unittest.TestCase):
    """
    Test pyspacer's Caffe extractor and compare to features extracted using
    CoralNet Beta's Caffe extractor.
    Tests pass if feature values are near identical.
    """

    def setUp(self):
        config.filter_warnings()

    def test_png(self):
        """
        Note that we use a png image here to avoid the inconsistencies
        with libjpeg versions.
        See discussion in https://github.com/beijbom/pyspacer/pull/10 for
        more details on libjpeg.
        """

        im_key = 's1388/i1023213'

        new_feats_loc = DataLocation(storage_type='memory',
                                     key='new_features.json')

        rowcols = [(1571, 1804)]

        msg = ExtractFeaturesMsg(
            job_token='beta_reg_test',
            feature_extractor_name='vgg16_coralnet_ver1',
            image_loc=cn_beta_fixture_location(im_key + '.png'),
            rowcols=rowcols,
            feature_loc=new_feats_loc
        )
        _ = extract_features(msg)

        legacy_feats = ImageFeatures.load(
            cn_beta_fixture_location(im_key + '.png.features.json')
        )

        self.assertFalse(legacy_feats.valid_rowcol)
        self.assertEqual(legacy_feats.npoints, len(rowcols))
        self.assertEqual(legacy_feats.feature_dim, 4096)

        new_feats = ImageFeatures.load(new_feats_loc)

        self.assertTrue(new_feats.valid_rowcol)
        self.assertEqual(new_feats.npoints, len(rowcols))
        self.assertEqual(new_feats.feature_dim, 4096)

        for legacy_pf, new_pf, rc in zip(legacy_feats.point_features,
                                         new_feats.point_features,
                                         msg.rowcols):
            self.assertTrue(np.allclose(legacy_pf.data, new_pf.data,
                                        atol=1e-5))


@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestClassifyFeatures(unittest.TestCase):
    """
    Get scores from the current classify_features task using previous
    scikit-learn versions' classifiers.
    Compare to scores calculated entirely with previous scikit-learn versions.
    Test pass if scores are near identical.
    """

    def setUp(self):
        config.filter_warnings()

    def run_one_test(
        self,
        feature_loc: DataLocation,
        classifier_loc: DataLocation,
        scores_loc: DataLocation
    ):

        msg = ClassifyFeaturesMsg(
            job_token='regression_test',
            feature_loc=feature_loc,
            classifier_loc=classifier_loc
        )
        new_return = classify_features(msg)

        legacy_return = ClassifyReturnMsg.load(scores_loc)

        with self.subTest(feats=feature_loc.key, clf=classifier_loc.key):
            for ls, ns in zip(legacy_return.scores, new_return.scores):
                self.assertTrue(np.allclose(ls[2], ns[2]))

    def test_cn_beta(self):
        for source, (clf, imgs) in cn_beta_fixtures.items():
            for img in imgs:
                self.run_one_test(
                    cn_beta_fixture_location(
                        f"{source}/{img}.features.json"),
                    cn_beta_fixture_location(
                        f"{source}/{clf}"),
                    cn_beta_fixture_location(
                        f"{source}/{img}.scores.json"),
                )

    def test_pyspacer_0_3_1_vgg16(self):
        for source_id, source_data in pyspacer_031_vgg16_fixtures.items():
            for image_id in source_data['images']:
                classifier_id = source_data['classifier']
                features_filename = f"{image_id}.featurevector"
                classifier_filename = f"{classifier_id}.model"
                scores_filename = \
                    f"img{image_id}_clf{classifier_id}.scores.json"
                self.run_one_test(
                    pyspacer031_vgg16_fixture_location(
                        f"s{source_id}/{features_filename}"),
                    pyspacer031_vgg16_fixture_location(
                        f"s{source_id}/{classifier_filename}"),
                    pyspacer031_vgg16_fixture_location(
                        f"s{source_id}/{scores_filename}"),
                )

    def test_pyspacer_0_3_1_efficientnet(self):
        for source_id, source_data in (
            pyspacer_031_efficientnet_fixtures.items()
        ):
            for image_id in source_data['images']:
                classifier_id = source_data['classifier']
                features_filename = f"{image_id}.featurevector"
                classifier_filename = f"{classifier_id}.model"
                scores_filename = \
                    f"img{image_id}_clf{classifier_id}.scores.json"
                self.run_one_test(
                    pyspacer031_efficientnet_fixture_location(
                        f"s{source_id}/{features_filename}"),
                    pyspacer031_efficientnet_fixture_location(
                        f"s{source_id}/{classifier_filename}"),
                    pyspacer031_efficientnet_fixture_location(
                        f"s{source_id}/{scores_filename}"),
                )


@unittest.skipUnless(config.HAS_CAFFE, 'Caffe not installed')
@unittest.skipUnless(config.HAS_S3_MODEL_ACCESS, 'No access to models')
@unittest.skipUnless(config.HAS_S3_TEST_ACCESS, 'No access to test bucket')
class TestExtractClassify(unittest.TestCase):
    """ Tests new feature extractor and a classification against legacy.
    Test passes if the same class is assigned in both cases for each
    row, col location, or if they differ but scores are close. """

    def setUp(self):
        config.filter_warnings()
        self.storage = storage_factory('s3', config.TEST_BUCKET)

    def test_tricky_example(self):
        """ From regression testing, this particular row, col location
        of this particular image gave the largest difference in
        classification scores """

        im_key = 's1388/i1023213'
        clf_key = 's1388/8942.model'

        rowcol = [(1571, 1804)]

        new_return, legacy_return = \
            extract_and_classify(im_key, clf_key, rowcol)

        for ls, ns in zip(legacy_return.scores, new_return.scores):
            legacy_pred = np.argmax(ls[2])
            new_pred = np.argmax(ns[2])

            score_diff_legacy_pred = np.abs(ns[2][legacy_pred] -
                                            ls[2][legacy_pred])

            score_diff_new_pred = np.abs(ns[2][new_pred] -
                                         ls[2][new_pred])

            # We pass the test of the predictions are identical.
            ok = legacy_pred == new_pred
            if not ok:

                # If prediction are not identical we still pass if the scores
                # are very similar.
                ok = score_diff_legacy_pred < 0.05 and \
                     score_diff_new_pred < 0.05

            self.assertTrue(ok)


if __name__ == '__main__':
    unittest.main()
