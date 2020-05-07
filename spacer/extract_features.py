"""
Defines feature-extractor ABC; implementations; and factory.
"""

import abc
import random
import time
from typing import List
from typing import Tuple

from PIL import Image

from spacer import config
from spacer.data_classes import PointFeatures, ImageFeatures
from spacer.extract_features_utils import crop_patches
from spacer.messages import ExtractFeaturesReturnMsg
from spacer.storage import download_model
from spacer.torch_utils import extract_feature


class FeatureExtractor(abc.ABC):  # pragma: no cover

    @abc.abstractmethod
    def __call__(self,
                 im: Image,
                 rowcols: List[Tuple[int, int]]) \
            -> Tuple[ImageFeatures, ExtractFeaturesReturnMsg]:
        """ Runs the feature extraction """

    @property
    @abc.abstractmethod
    def feature_dim(self) -> int:
        """ Returns the feature dimension of extractor. """


class DummyExtractor(FeatureExtractor):
    """
    This doesn't actually extract any features from the image,
    it just returns dummy information.
    Note that feature dimension is compatible with the VGG16CaffeExtractor.
    """
    def __init__(self, feature_dim):
        self._feature_dim = feature_dim

    def __call__(self, im, rowcols):
        return ImageFeatures(
            point_features=[PointFeatures(row=rc[0],
                                          col=rc[1],
                                          data=[random.random() for _ in
                                                range(self.feature_dim)])
                            for rc in rowcols],
            valid_rowcol=True,
            npoints=len(rowcols),
            feature_dim=self.feature_dim
        ), ExtractFeaturesReturnMsg.example()

    @property
    def feature_dim(self):
        return self._feature_dim


class VGG16CaffeExtractor(FeatureExtractor):

    def __init__(self):

        # Cache models and prototxt locally.
        self.modeldef_path, _ = download_model(
            'vgg16_coralnet_ver1.deploy.prototxt')
        self.modelweighs_path, self.model_was_cashed = download_model(
            'vgg16_coralnet_ver1.caffemodel')

    def __call__(self, im, rowcols):

        # We should only reach this line if it is confirmed caffe is available
        from spacer.caffe_utils import classify_from_patchlist

        start_time = time.time()

        # Set caffe parameters
        caffe_params = {'im_mean': [128, 128, 128],
                        'scaling_method': 'scale',
                        'crop_size': 224,
                        'batch_size': 10}

        # Crop patches
        patch_list = crop_patches(im, rowcols, caffe_params['crop_size'])
        del im

        # Extract features
        feats = classify_from_patchlist(patch_list,
                                        caffe_params,
                                        self.modeldef_path,
                                        self.modelweighs_path,
                                        scorelayer='fc7')

        return \
            ImageFeatures(
                point_features=[PointFeatures(row=rc[0],
                                              col=rc[1],
                                              data=ft.tolist())
                                for rc, ft in zip(rowcols, feats)],
                valid_rowcol=True,
                feature_dim=len(feats[0]),
                npoints=len(feats)
            ), ExtractFeaturesReturnMsg(
                model_was_cashed=self.model_was_cashed,
                runtime=time.time() - start_time
            )

    @property
    def feature_dim(self):
        return 4096


class EfficientNetExtractor(FeatureExtractor):

    def __init__(self):

        # Cache models locally.
        self.modelweighs_path, self.model_was_cashed = download_model(
            'efficientnetb0_5eps_best.pt')

    def __call__(self, im, rowcols):

        start_time = time.time()

        # Set torch parameters
        torch_params = {'model_type': 'efficientnet',
                        'model_name': 'efficientnet-b0',
                        'weights_path': self.modelweighs_path,
                        'num_class': 1279,
                        'crop_size': 224,
                        'batch_size': 10}

        # Crop patches
        patch_list = crop_patches(im, rowcols, torch_params['crop_size'])
        del im

        # Extract features
        feats = extract_feature(patch_list, torch_params)

        return ImageFeatures(
            point_features=[PointFeatures(row=rc[0], col=rc[1], data=ft)
                            for rc, ft in zip(rowcols, feats)],
            valid_rowcol=True, feature_dim=len(feats[0]), npoints=len(feats)
        ), ExtractFeaturesReturnMsg(
            model_was_cashed=self.model_was_cashed,
            runtime=time.time() - start_time
        )

    @property
    def feature_dim(self):
        return 1280


def feature_extractor_factory(modelname,
                              dummy_featuredim=4096) -> FeatureExtractor:

    assert modelname in config.FEATURE_EXTRACTOR_NAMES, \
        "Model name {} not registered".format(modelname)

    if modelname == 'vgg16_coralnet_ver1':
        assert config.HAS_CAFFE, \
            "Need Caffe installed to instantiate {}".format(modelname)
        print("-> Initializing VGG16CaffeExtractor")
        return VGG16CaffeExtractor()
    if modelname == 'efficientnet_b0_ver1':
        print("-> Initializing EfficientNetExtractor")
        return EfficientNetExtractor()
    if modelname == 'dummy':
        print("-> Initializing DummyExtractor")
        return DummyExtractor(dummy_featuredim)
