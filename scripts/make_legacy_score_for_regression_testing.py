"""
This script was used to create fixture for regression testing.
Legacy models on CoralNet were loaded up using scikit-learn 0.17.
The models were then used to classify features and the scores were stored.
This allows unit-tests to be created to make sure that when classifiers
are loaded using newer versions of sci-kit learn the predicted scores
are compatible.
"""

import os.path as osp
from spacer.messages import \
    ClassifyFeaturesMsg, \
    DataLocation

from spacer.data_classes import ImageFeatures

from spacer.tasks import classify_features

import warnings

import pkg_resources


reg_meta = {
    's16': ('1355.model', ['i2921', 'i2934']),
    's295': ('10279.model', ['i1370227', 'i160100']),
    's603': ('3709.model', ['i576858', 'i576912']),
    's812': ('4772.model', ['i672762', 'i674185']),
    's1388': ('8942.model', ['i1023182', 'i1023213'])
}


def main():

    # Note, we uploaded these to S3, so this path is temporary
    fixtures_root = '/Users/beijbom/Desktop/beta_reg/'

    # Make sure this code will not run if there are any UserWarnings
    # This is what scikit-learn >= 0.18 will raise when trying to unpickle.
    warnings.simplefilter("error", UserWarning)

    assert pkg_resources.get_distribution("scikit-learn").version == '0.17.1',\
        "Must use scikit-learn 0.17.1 to run this script."

    for source_name in reg_meta:

        clf_name = reg_meta[source_name][0]
        for img_prefix in reg_meta[source_name][1]:

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


def tmp():
    import numpy as np
    new_feats_loc = DataLocation(
        storage_type='filesystem',
        key='/Users/beijbom/spacermodels2/i160100.features.json'
    )
    old_feats_loc = DataLocation(
        storage_type='filesystem',
        key='/Users/beijbom/pyspacer/spacer/tests/fixtures/s295/i160100.features.json'
    )
   # ? old_feats_loc = DataLocation(
   #      storage_type='filesystem',
   #      key='/Users/beijbom/spacermodels2/alqyt4nl6y.JPG.featurevector'
   #  )


    new_feats = ImageFeatures.load(new_feats_loc)
    old_feats = ImageFeatures.load(old_feats_loc)

    for pf_new in new_feats.point_features:
        dn = pf_new.data
        diffs = []
        for pf in old_feats.point_features:
            do = pf.data
            diffs.append(np.max(np.abs(np.array(dn) - np.array(do))))
        print(diffs)



if __name__ == '__main__':
    # tmp()
    main()
