from spacer import config


# Extractors used in unit tests.
#
# Here we define the extractors in serialized form so that the S3 DataLocations
# aren't instantiated when importing this module. (Instantiation would get an
# error if the S3 bucket isn't available.)
TEST_EXTRACTORS = {
    'vgg16': dict(
        class_path='spacer.extract_features.VGG16CaffeExtractor',
        data_locations=dict(
            definition=dict(
                storage_type='s3',
                key='vgg16_coralnet_ver1.deploy.prototxt',
                bucket_name=config.TEST_EXTRACTORS_BUCKET,
            ),
            weights=dict(
                storage_type='s3',
                key='vgg16_coralnet_ver1.caffemodel',
                bucket_name=config.TEST_EXTRACTORS_BUCKET,
            ),
        ),
        data_hashes=dict(
            weights='fb83781de0e207ded23bd42d7eb6e75c1e915a6fbef74120f72732984e227cca',
        ),
    ),
    'efficientnet-b0': dict(
        class_path='spacer.extract_features.EfficientNetExtractor',
        data_locations=dict(
            weights=dict(
                storage_type='s3',
                key='efficientnet_b0_ver1.pt',
                bucket_name=config.TEST_EXTRACTORS_BUCKET,
            ),
        ),
        data_hashes=dict(
            weights='c3dc6d304179c6729c0a0b3d4e60c728bdcf0d82687deeba54af71827467204c',
        ),
    ),
}
