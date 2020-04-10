
from spacer.messages import DataLocation, ExtractFeaturesMsg
from spacer.tasks import extract_features


def run():

    msg = ExtractFeaturesMsg(
        job_token='beta_reg_test',
        feature_extractor_name='vgg16_coralnet_ver1',
        image_loc=DataLocation(storage_type='s3',
                               bucket_name='spacer-test',
                               key='beta_reg/s1388/i1023182.jpg'),
        rowcols=[(2819, 1503)],
        feature_loc=DataLocation(storage_type='s3',
                                 bucket_name='spacer-test',
                                 key='tmp/new/i1023182.feats')
    )
    _ = extract_features(msg)


if __name__ == '__main__':
    run()
