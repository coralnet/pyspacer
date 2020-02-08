import abc
import time
import os
from typing import Tuple, List
from spacer.messages import ExtractFeaturesMsg, ExtractFeaturesReturnMsg
from spacer import config
from spacer.storage import Storage, download_model


class FeatureExtractorInterface(abc.ABC):

    @abc.abstractmethod
    def __call__(self) -> Tuple[List[List[float]], ExtractFeaturesReturnMsg]:
        pass


class VGG16CaffeExtractor(FeatureExtractorInterface):

    def __init__(self, msg: ExtractFeaturesMsg, storage: Storage):
        self.payload = msg
        self.storage = storage

    def __call__(self, *args, **kwargs):

        # We should only reach this line if it is confirmed caffe is available
        os.environ['GLOG_minloglevel'] = '2'
        import caffe
        from spacer.caffe_backend.utils import classify_from_patchlist

        t1 = time.time()

        # Make sure the right model and prototxt are available locally.
        modeldef_path, def_was_cashed = download_model(self.payload.modelname + '.deploy.prototxt')
        modelweighs_path, weights_was_cashed = download_model(self.payload.modelname + '.caffemodel')
        was_cashed = def_was_cashed and weights_was_cashed

        # Setup caffe
        caffe.set_mode_cpu()
        net = caffe.Net(modeldef_path, modelweighs_path, caffe.TEST)

        # Set parameters
        pyparams = {'im_mean': [128, 128, 128],
                    'scaling_method': 'scale',
                    'scaling_factor': 1,
                    'crop_size': 224,
                    'batch_size': 10}

        dummy_label = 1  # The imdict data structure needs a label, but we don't care what it is.
        rowcols = [(row, col, dummy_label) for row, col in self.payload.rowcols]
        imdict = {
            self.payload.imkey: (rowcols, 100)
        }

        # Run
        t2 = time.time()
        (_, _, feats) = classify_from_patchlist(imdict, pyparams, net, self.storage, scorelayer='fc7')
        feats = [f.tolist() for f in feats]

        return_message = ExtractFeaturesReturnMsg(
            model_was_cashed=was_cashed,
            runtime={
                'total': time.time() - t1,
                'core': time.time() - t2,
                'per_point': (time.time() - t2) / len(self.payload.rowcols)
            }
        )

        return feats, return_message


class EfficientNetExtractor(FeatureExtractorInterface):

    def __init__(self, msg: ExtractFeaturesMsg, storage: Storage):
        self.msg = msg
        self.storage = storage

    def __call__(self, *args, **kwargs):
        pass


def feature_extractor_factory(msg: ExtractFeaturesMsg, storage: Storage) -> FeatureExtractorInterface:

    assert msg.modelname in config.FEATURE_EXTRACTOR_NAMES, "Model name {} not registered".format(msg.modelname)

    if msg.modelname == 'vgg16_coralnet_ver1':
        assert config.HAS_CAFFE, "Need to have Caffe installed to instantiate {}".format(msg.modelname)
        print("-> Initializing VGG16CaffeExtractor")
        return VGG16CaffeExtractor(msg, storage)

    elif msg.modelname == 'efficientnet_b0_imagenet':
        print("-> Initializing EfficientNetExtractor")
        return EfficientNetExtractor(msg, storage)

    else:
        raise ValueError('Unknown modelname: {}'.format(msg.modelname))