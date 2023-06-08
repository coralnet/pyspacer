"""
This file has scripts to run regression testing against coralnet beta.

The build method does not need to be run again,
it was used to create the tests as follows:
Legacy models on CoralNet were loaded up using scikit-learn 0.17.
The models were then used to classify features and the scores were stored.
This allows unit-tests to be created to make sure that when classifiers
are loaded using newer versions of sci-kit learn the predicted scores
are compatible.

The run method runs the regression tests. This test
runs both feature extraction and classification.
Since the libjpeg version was changed the results will not be identical.
However, the scores should be close. See discussion in
(https://github.com/beijbom/pyspacer/pull/10)
for more details.

We also run a subset of these tests as part of the standard test suite.
"""

import json
import os.path as osp
import numpy as np
from spacer.messages import \
    ClassifyFeaturesMsg, \
    DataLocation

from spacer import config

from spacer.tasks import classify_features
from spacer.storage import storage_factory
from spacer.tests.test_legacy import cn_beta_fixtures, extract_and_classify
from spacer.tests.utils import cn_beta_fixture_location

import pkg_resources


def build():

    # Note, we uploaded these to S3, so this path is temporary
    fixtures_root = '/Users/beijbom/Desktop/beta_reg/'

    config.filter_warnings()

    assert pkg_resources.get_distribution("scikit-learn").version == '0.17.1',\
        "Must use scikit-learn 0.17.1 to run this script."

    for source_name in cn_beta_fixtures:

        clf_name = cn_beta_fixtures[source_name][0]
        for img_prefix in cn_beta_fixtures[source_name][1]:

            msg = ClassifyFeaturesMsg(
                job_token='regression_test',
                feature_loc=DataLocation(storage_type='filesystem',
                                         key=osp.join(fixtures_root,
                                                      source_name,
                                                      img_prefix +
                                                      '.features.json')),
                classifier_loc=DataLocation(storage_type='filesystem',
                                            key=osp.join(fixtures_root,
                                                         source_name,
                                                         clf_name))
            )
            return_msg = classify_features(msg)
            score_path = osp.join(fixtures_root,
                                  source_name,
                                  img_prefix + '.scores.json')
            score_loc = DataLocation(storage_type='filesystem',
                                     key=score_path)
            return_msg.store(score_loc)
            print('Wrote scores to: {}'.format(score_path))


def log(msg):
    with open('/workspace/models/beta_regression.log', 'a') as f:
        f.write(msg + '\n')
        print(msg)


def get_rowcol(key, storage):
    """ This file was saved using
    coralnet/project/vision_backend/management/commands/
    vb_export_spacer_data.py

    https://github.com/beijbom/coralnet/blob/
    e08afaa0164425fc16ae4ed60841d70f2eff59a6/project/vision_backend/
    management/commands/vb_export_spacer_data.py
    """
    anns = json.loads(storage.load(key).getvalue().decode('utf-8'))
    return [(entry['row'], entry['col']) for entry in anns]


def run():

    def run_one_test(im_key, clf_key):

        rowcol = get_rowcol(
            cn_beta_fixture_location(im_key + '.anns.json'), storage)

        new_return, legacy_return = \
            extract_and_classify(im_key, clf_key, rowcol)

        for ls, ns, rc in zip(legacy_return.scores, new_return.scores, rowcol):
            legacy_pred = np.argmax(ls[2])
            new_pred = np.argmax(ns[2])

            score_diff_legacy_pred = np.abs(ns[2][legacy_pred] -
                                            ls[2][legacy_pred])

            score_diff_new_pred = np.abs(ns[2][new_pred] -
                                         ls[2][new_pred])

            # We pass the test of the predictions are identical.
            all_score_diffs.append(score_diff_legacy_pred)
            all_score_diffs.append(score_diff_new_pred)
            ok = legacy_pred == new_pred
            if not ok:
                # If prediction are not identical we still pass if the scores
                # are very similar.
                ok = score_diff_legacy_pred < 0.05 and \
                     score_diff_new_pred < 0.05

            if ok:
                log("{} [{}] passed with new_pred={}, legacy_pred={}, "
                    "score_diff_legacy_pred={:.3f}, score_diff_new_pred={:.3f}".
                    format(im_key, rc, new_pred, legacy_pred,
                           score_diff_legacy_pred, score_diff_new_pred))
            else:
                log("!!!!!! {} [{}] failed with new_pred={}, legacy_pred={}, "
                    "score_diff_legacy_pred={:.3f}, score_diff_new_pred={:.3f}"
                    " !!!!!!".
                    format(im_key, rc, new_pred, legacy_pred,
                           score_diff_legacy_pred, score_diff_new_pred))

    assert config.HAS_CAFFE, "Must have caffe installed to run the reg tests."

    storage = storage_factory('s3', config.TEST_BUCKET)

    all_score_diffs = []

    for source, (clf, imgs) in cn_beta_fixtures.items():
        for img in imgs:
            run_one_test(source + '/' + img, source + '/' + clf)

    log("Max score diff: {}".format(np.max(all_score_diffs)))


if __name__ == '__main__':
    run()
