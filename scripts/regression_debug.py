import numpy as np
from spacer.data_classes import ImageFeatures
from spacer.messages import DataLocation


def norm_diff(imfeat1, imfeat2):
    norms = []
    for pf1, pf2 in zip(imfeat1.point_features, imfeat2.point_features):
        norms.append(np.linalg.norm(np.array(pf1.data) - np.array(pf2.data)))
    return norms


leg_feats = ImageFeatures.load(DataLocation(
    storage_type='s3',
    key='tmp/legacy/i1023182.feats',
    bucket_name='spacer-test'
))


prod_feats = ImageFeatures.load(DataLocation(
    storage_type='s3',
    key='beta_reg/s1388/i1023182.features.json',
    bucket_name='spacer-test'
))

new_feats = ImageFeatures.load(DataLocation(
    storage_type='s3',
    key='tmp/new/i1023182.feats',
    bucket_name='spacer-test'
))

print('leg vs prod: {}'.format(norm_diff(leg_feats, prod_feats)))
print('new vs prod: {}'.format(norm_diff(new_feats, prod_feats)))
