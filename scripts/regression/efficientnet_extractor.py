"""
This file is used for:
    1. run feature extractor
    2. train the classifier
on the exported spacer training data and confirms that
it get similar (or higher) performance than production.

To use do:
python scripts/regression/efficientnet_extractor.py \
    efficientnet_b0_ver1 294 10 MLP /path/to/features
"""

import os
import fire

from scripts.regression.utils import build_traindata, do_training, cache_local, \
    do_extract_features


def main(extractor_name: str,
         source_id: int,
         n_epochs: int,
         clf_type: str,
         local_path: str) -> None:

    source_root = os.path.join(local_path, 's{}'.format(source_id))
    image_root = os.path.join(source_root, 'images')

    # Download image, metadata and labels for local feature extraction
    cache_local(source_root, image_root, 'beta_export', source_id,
                cache_image=True, cache_feats=False)

    # Extract features
    do_extract_features(extractor_name, image_root)

    # Create the train and val ImageLabels data structures.
    train_labels, val_labels = build_traindata(image_root)

    # Perform training
    do_training(source_root, train_labels, val_labels, n_epochs, clf_type)


if __name__ == '__main__':
    fire.Fire(main)
