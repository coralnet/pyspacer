import abc
import os
import time
from typing import Tuple

from spacer import config
from spacer.messages import ExtractFeaturesMsg, ExtractFeaturesReturnMsg, \
    ImageFeatures, PointFeatures
from spacer.storage import Storage, download_model


class FeatureExtractorInterface(abc.ABC):

    @abc.abstractmethod
    def __call__(self) -> Tuple[ImageFeatures, ExtractFeaturesReturnMsg]:
        pass


class DummyExtractor(FeatureExtractorInterface):
    """
    This doesn't actually extract any features from the image,
    it just returns dummy information.
    """

    def __init__(self, msg: ExtractFeaturesMsg, storage: Storage):
        self.msg = msg
        self.storage = storage

    def __call__(self, *args, **kwargs):
        img_features = ImageFeatures(
            point_features=[PointFeatures(row=rc[0],
                                          col=rc[1],
                                          data=[1.1, 2.2, 3.3])
                            for rc in self.msg.rowcols],
            valid_rowcol=True,
            npoints=len(self.msg.rowcols),
            feature_dim=3
        )
        return img_features, ExtractFeaturesReturnMsg.example()


class VGG16CaffeExtractor(FeatureExtractorInterface):

    def __init__(self, msg: ExtractFeaturesMsg, storage: Storage):
        self.payload = msg
        self.storage = storage

    def __call__(self, *args, **kwargs):

        # We should only reach this line if it is confirmed caffe is available
        # This suppresses most of the superfluous caffe logging.
        os.environ['GLOG_minloglevel'] = '3'
        from spacer.caffe_backend.utils import classify_from_patchlist

        t1 = time.time()

        # Make sure the right model and prototxt are available locally.
        modeldef_path, _ = download_model(
            self.payload.modelname + '.deploy.prototxt')
        modelweighs_path, was_cashed = download_model(
            self.payload.modelname + '.caffemodel')

        # Set parameters
        pyparams = {'im_mean': [128, 128, 128],
                    'scaling_method': 'scale',
                    'scaling_factor': 1,
                    'crop_size': 224,
                    'batch_size': 10}

        # The imdict data structure needs a label, it's not used.
        dummy_label = 1
        # The imheight in centimeters are not used by the algorithm.
        dummy_imheight = 100
        rowcols = [(row, col, dummy_label) for row, col in self.payload.rowcols]
        imdict = {self.payload.imkey: (rowcols, dummy_imheight)}

        # Run
        t2 = time.time()
        (_, _, feats) = classify_from_patchlist(imdict,
                                                pyparams,
                                                modeldef_path,
                                                modelweighs_path,
                                                self.storage,
                                                scorelayer='fc7')

        img_features = ImageFeatures(
            point_features=[PointFeatures(row=rc[0],
                                          col=rc[1],
                                          data=ft.tolist())
                            for rc, ft in zip(rowcols, feats)],
            valid_rowcol=True,
            feature_dim=len(feats[0]),
            npoints=len(feats)
        )

        return_message = ExtractFeaturesReturnMsg(
            model_was_cashed=was_cashed,
            runtime={
                'total': time.time() - t1,
                'core': time.time() - t2,
                'per_point': (time.time() - t2) / len(self.payload.rowcols)
            }
        )

        return img_features, return_message


class EfficientNetExtractor(FeatureExtractorInterface):

    def __init__(self, msg: ExtractFeaturesMsg, storage: Storage):
        self.msg = msg
        self.storage = storage

    def __call__(self, *args, **kwargs):
        pass


def feature_extractor_factory(msg: ExtractFeaturesMsg, storage: Storage) -> FeatureExtractorInterface:

    assert msg.modelname in config.FEATURE_EXTRACTOR_NAMES, \
        "Model name {} not registered".format(msg.modelname)

    if msg.modelname == 'vgg16_coralnet_ver1':
        assert config.HAS_CAFFE, \
            "Need Caffe installed to instantiate {}".format(msg.modelname)
        print("-> Initializing VGG16CaffeExtractor")
        return VGG16CaffeExtractor(msg, storage)
    elif msg.modelname == 'efficientnet_b0_imagenet':
        print("-> Initializing EfficientNetExtractor")
        return EfficientNetExtractor(msg, storage)
    elif msg.modelname == 'dummy':
        print("-> Initializing DummyExtractor")
        return DummyExtractor(msg, storage)
    else:
        raise ValueError('Unknown modelname: {}'.format(msg.modelname))
