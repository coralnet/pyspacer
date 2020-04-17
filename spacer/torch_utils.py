"""
This file contains a set of pytorch utility functions
"""

import torch
import models
import numpy as np
from torchvision import transforms
from collections import OrderedDict


def gray2rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret


def transformation():
    """
    Transform an image or numpy array and normalize to [0, 1]
    :return: a transformer which takes in a image and return a normalized tensor
    """

    transformer = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transformer


def crop_patch(im, crop_size, scale, point_anns, offsets=None):
    """
    Crop patches from an image
    :param im: image for cropping
    :param crop_size: patch size
    :param scale: only supports scale = 1
    :param point_anns: [(row1, col1, 1), (row2, col2, 1), ...]
    :param offsets: offset to the original (row, col)
    :return: patch list and label list
    """

    assert scale == 1, "Only supports scale == 1"

    if offsets is None:
        offsets = np.zeros([len(point_anns), 2])

    patchlist = []
    labellist = []
    pad = crop_size

    im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    for ((row, col, label), offset) in zip(point_anns, offsets):
        center_org = np.asarray([row, col])
        center = np.round(pad + (center_org * scale) + offset).astype(np.int)

        patch = crop_simple(im, center, crop_size)

        patchlist.append(patch)
        labellist.append(label)

    return patchlist, labellist


def crop_simple(im, center, crop_size):
    """ Crops an image around the given center. """
    upper = int(center[0] - crop_size / 2)
    left = int(center[1] - crop_size / 2)
    return im[upper: upper + crop_size, left: left + crop_size, :]


def load_weights(model, modelweighs_path):
    """
    Load model weights, original weight saved with DataParallel
    Create new OrderedDict that does not contain `module`.
    :param model: Currently support EfficientNet
    :param modelweighs_path: pretrained model weight from new CoralNet
    :return: well trained model
    """
    state_dicts = torch.load(modelweighs_path, map_location=torch.device('cpu'))
    new_state_dicts = OrderedDict()
    for k, v in state_dicts['net'].items():
        name = k[7:]
        new_state_dicts[name] = v
    model.load_state_dict(new_state_dicts)
    for param in model.parameters():
        param.requires_grad = False
    return model


def classify_from_patchlist(im_pil, point_anns, pyparams):
    """
    Crop patches and extract features
    :param im_pil: Image
    :param point_anns: List[Tuple[int, int, int]] -> List[Tuple(row, col, label)]
    :param pyparams: parameter dict
    :return:
    """
    # Model setup and load pretrained weight
    net = models.get_model(pyparams['model_type'], pyparams['model_name'])
    net = load_weights(net, pyparams['weights_path'])

    scale = 1
    feats_list, gtlist = [], []
    transformer = transformation()

    # Convert to numpy (call 2 times needed otherwise get a
    # "IndexError: tuple index out of range" error. No idea why!
    _ = np.asarray(im_pil)
    im = np.asarray(im_pil)

    if len(im.shape) == 2 or im.shape[2] == 1:
        im = gray2rgb(im)
    im = im[:, :, :3]  # only keep the first three color channels

    # Crop patches
    patch_list, label_list = crop_patch(im, pyparams['crop_size'], scale, point_anns)
    patch_list = torch.stack([transformer(i) for i in patch_list])

    # Feed forward and extract features
    num_batch = int(np.ceil(len(patch_list) / pyparams['batch_size']))
    bs = pyparams['batch_size']
    for b in range(num_batch):
        current_batch = patch_list[b*bs: b*bs + min(len(patch_list[b*bs:]), bs)]
        features = net.extract_features(current_batch)
        feats_list.extend(features.tolist())

    return label_list, feats_list
