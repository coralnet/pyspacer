"""
These tests check for compatibility with results of previous coralnet/pyspacer
versions.
Much of the test data was prepared using scripts/regression/caffe_extractor.py
"""
import unittest

import numpy as np

from spacer import config
from spacer.data_classes import DataLocation, ImageFeatures
from spacer.extractors import FeatureExtractor
from spacer.messages import (
    ClassifyFeaturesMsg,
    ClassifyReturnMsg,
    ExtractFeaturesMsg,
)
from spacer.tasks import classify_features, extract_features
from spacer.tests.utils import cn_beta_fixture_location
from .common import TEST_EXTRACTORS
from .decorators import \
    require_caffe, require_cn_fixtures, require_cn_test_extractors


cn_beta_fixtures = {
    's16': ('1355.model', ['i2921', 'i2934']),
    's295': ('10279.model', ['i1370227', 'i160100']),
    's603': ('3709.model', ['i576858', 'i576912']),
    's812': ('4772.model', ['i672762', 'i674185']),
    's1388': ('8942.model', ['i1023182', 'i1023213'])
}
pyspacer_fixtures = {
    's23': ('test.model', ['IMG_4619', 'IMG_4700']),
    's1288': ('test.model', ['ROS-03_2018_A_30', 'ROS-06_2018_A_30']),
}


def pyspacer_fixture_location(version, extractor, key):
    return DataLocation(
        storage_type='s3',
        bucket_name=config.CN_FIXTURES_BUCKET,
        key=f'legacy_compat/pyspacer_{version}/{extractor}/{key}'
    )


def extract_and_classify(im_key, clf_key, rowcol):
    """ Helper method for extract_and_classify regression tests. """
    new_feats_loc = DataLocation(storage_type='memory',
                                 key='features.json')

    msg = ExtractFeaturesMsg(
        job_token='beta_reg_test',
        extractor=FeatureExtractor.deserialize(TEST_EXTRACTORS['vgg16']),
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


@require_caffe
@require_cn_test_extractors
@require_cn_fixtures
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
        See discussion in https://github.com/coralnet/pyspacer/pull/10 for
        more details on libjpeg.
        """

        im_key = 's1388/i1023213'

        new_feats_loc = DataLocation(storage_type='memory',
                                     key='new_features.json')

        rowcols = [(1571, 1804)]

        msg = ExtractFeaturesMsg(
            job_token='beta_reg_test',
            extractor=FeatureExtractor.deserialize(TEST_EXTRACTORS['vgg16']),
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


@require_cn_fixtures
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

    def do_test_pyspacer(self, version, extractor):
        for source, (clf, imgs) in pyspacer_fixtures.items():
            for img in imgs:
                self.run_one_test(
                    pyspacer_fixture_location(
                        version, extractor, f"{source}/{img}.featurevector"),
                    pyspacer_fixture_location(
                        version, extractor, f"{source}/{clf}"),
                    pyspacer_fixture_location(
                        version, extractor, f"{source}/{img}.scores.json"),
                )

    def test_pyspacer_0_3_1_vgg16(self):
        self.do_test_pyspacer('0.3.1', 'vgg16')

    def test_pyspacer_0_3_1_efficientnet(self):
        self.do_test_pyspacer('0.3.1', 'efficientnet')

    def test_pyspacer_0_9_0_vgg16(self):
        self.do_test_pyspacer('0.9.0', 'vgg16')

    def test_pyspacer_0_9_0_efficientnet(self):
        self.do_test_pyspacer('0.9.0', 'efficientnet')


@require_caffe
@require_cn_test_extractors
@require_cn_fixtures
class TestExtractClassify(unittest.TestCase):
    """ Tests new feature extractor and a classification against legacy.
    Test passes if the same class is assigned in both cases for each
    row, col location, or if they differ but scores are close. """

    def setUp(self):
        config.filter_warnings()

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
