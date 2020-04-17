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
from spacer.messages import ExtractFeaturesReturnMsg
from spacer.storage import download_model


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

        # The imdict data structure needs a label, set to 1, it's not used.
        rowcollabels = [(row, col, 1) for row, col in rowcols]

        (_, _, feats) = classify_from_patchlist(im,
                                                rowcollabels,
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

    def __call__(self, im, rowcols):
        raise NotImplementedError

    @property
    def feature_dim(self):
        raise NotImplementedError


def feature_extractor_factory(modelname,
                              dummy_featuredim=4096) -> FeatureExtractor:

    assert modelname in config.FEATURE_EXTRACTOR_NAMES, \
        "Model name {} not registered".format(modelname)

    if modelname == 'vgg16_coralnet_ver1':
        assert config.HAS_CAFFE, \
            "Need Caffe installed to instantiate {}".format(modelname)
        print("-> Initializing VGG16CaffeExtractor")
        return VGG16CaffeExtractor()
    if modelname == 'efficientnet_b0_imagenet':
        print("-> Initializing EfficientNetExtractor")
        return EfficientNetExtractor()
    if modelname == 'dummy':
        print("-> Initializing DummyExtractor")
        return DummyExtractor(dummy_featuredim)
