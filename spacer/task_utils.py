from __future__ import annotations
from logging import getLogger

from PIL import Image

from spacer import config
from spacer.data_classes import ImageLabels
from spacer.exceptions import (
    DataLimitError, RowColumnInvalidError, TrainingLabelsError)
from spacer.messages import TrainingTaskLabels

logger = getLogger(__name__)


def check_extract_inputs(image: Image,
                         rowcols: list[tuple[int, int]],
                         im_key: str):

    im_width, im_height = image.size

    if im_width * im_height > config.MAX_IMAGE_PIXELS:
        raise DataLimitError(
            f"Image {im_key} has {im_width} x {im_height}"
            f" = {im_width * im_height} total pixels, which is larger"
            f" than the max allowed of {config.MAX_IMAGE_PIXELS}."
        )
    if len(rowcols) > config.MAX_POINTS_PER_IMAGE:
        raise DataLimitError(
            f"{len(rowcols)} point locations were specified for image"
            f" {im_key}, and that's larger than the max allowed of"
            f" {config.MAX_POINTS_PER_IMAGE}."
        )

    for row, col in rowcols:

        if not isinstance(row, int):
            raise RowColumnInvalidError(
                f"{im_key}: Row values must be integers."
                f" Given value was: {row}")
        if not isinstance(col, int):
            raise RowColumnInvalidError(
                f"{im_key}: Column values must be integers."
                f" Given value was: {col}")

        if row < 0:
            raise RowColumnInvalidError(
                f"{im_key}: Row values must be non-negative."
                f" Given value was: {row}")
        if col < 0:
            raise RowColumnInvalidError(
                f"{im_key}: Column values must be non-negative."
                f" Given value was: {col}")

        if row >= im_height:
            raise RowColumnInvalidError(
                f"{im_key}: Row value {row} falls outside this image's"
                f" valid range of 0-{im_height - 1}.")
        if col >= im_width:
            raise RowColumnInvalidError(
                f"{im_key}: Column value {col} falls outside this image's"
                f" valid range of 0-{im_width - 1}.")


def preprocess_labels(
        # Training labels (annotations); two acceptable formats:
        #
        # 1) An ImageLabels instance, which this function decides how to
        # split up into training, reference, and validation sets.
        # 2) A dict which has keys 'train', 'ref', and 'val',
        # each mapping to a separate ImageLabels instance.
        labels_in: ImageLabels | TrainingTaskLabels,
        # If present, the passed labels are filtered so that any classes not
        # included in this set have their labels discarded.
        accepted_classes: set[int] | None = None) -> TrainingTaskLabels:
    """
    This function can be used to preprocess labels before creating a
    TrainClassifierMsg. It does:
    1) Splitting of labels into train/ref/val, if not already done
    2) Filtering of labels by valid_classes, if that arg is specified
    3) Error checks

    This is also called by pyspacer after receiving a TrainClassifierMsg
    to ensure the error checks are done.
    """

    if isinstance(labels_in, TrainingTaskLabels):
        # The caller has decided how to split the data into
        # training, reference, and validation sets.
        labels = labels_in
    else:
        # Split data into training, reference, and validation sets.
        #
        # Arbitrarily, validation gets 10%, reference gets
        # min(10%, TRAINING_BATCH_LABEL_COUNT), training gets the rest.
        # This is imprecise because it's split on the image level, not the
        # label level, and images can have different numbers of labels.
        #
        # The split is done in a way which guarantees that all 3 sets are
        # non-empty if there are at least 3 images.
        if len(labels_in) < 3:
            raise TrainingLabelsError(
                f"The training data has {len(labels_in)} image(s),"
                f" but need at least 3 to populate train/ref/val sets.")

        train_data = dict()
        ref_data = dict()
        val_data = dict()
        ref_label_count = 0
        ref_done = False

        for image_index, image_key in enumerate(labels_in.image_keys):
            this_image_labels = labels_in[image_key]

            if image_index % 10 == 0:
                # 1st, 11th, 21st, etc. images go in val.
                val_data[image_key] = this_image_labels
            elif not ref_done and image_index % 10 == 1:
                # 2nd, 12th, 22nd, etc. images go in ref, if ref still
                # has room within the batch size; else, go in train.
                if (ref_label_count + len(this_image_labels)
                        <= config.TRAINING_BATCH_LABEL_COUNT):
                    ref_data[image_key] = this_image_labels
                    ref_label_count += len(this_image_labels)
                else:
                    # ref would go over the batch size if it added this
                    # image.
                    train_data[image_key] = this_image_labels
                    ref_done = True
            else:
                # The rest go in train.
                train_data[image_key] = this_image_labels

        labels = TrainingTaskLabels(
            train=ImageLabels(train_data),
            ref=ImageLabels(ref_data),
            val=ImageLabels(val_data),
        )

    # Identify classes common to both train and ref.
    train_classes = labels.train.classes_set
    ref_classes = labels.ref.classes_set
    train_ref_classes = train_classes.intersection(ref_classes)

    # Further filter the classes if this arg was specified.
    if accepted_classes:
        classes_filter = \
            train_ref_classes.intersection(accepted_classes)
    else:
        classes_filter = train_ref_classes

    if len(classes_filter) <= 1:
        raise TrainingLabelsError(
            f"Need multiple classes to do training."
            f" After preprocessing training data, there are"
            f" {len(classes_filter)} class(es) left.")

    for set_name in ['train', 'ref', 'val']:
        labels[set_name] = \
            labels[set_name].filter_classes(classes_filter)

        if len(labels[set_name]) == 0:
            raise TrainingLabelsError(
                f"After preprocessing training data,"
                f" '{set_name}' set is empty.")

    return labels
