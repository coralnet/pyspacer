from __future__ import annotations
from collections import defaultdict
from enum import Enum
from logging import getLogger

from PIL import Image
from sklearn.model_selection import train_test_split

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


class ClassSamplingMethod(Enum):
    """
    How to consider classes when splitting annotations between
    train, ref, and val sets.
    """
    # Sample with no regard to classes.
    IGNORE = 'ignore'
    # Stratified sampling by class: an A%/B%/C% train/ref/val split means
    # about an A%/B%/C% split of each class.
    #
    # Stratification checks that the number of annotations in each
    # set isn't less than the number of unique classes.
    # However, each set is NOT guaranteed to have at least 1 of each class.
    # If stratification is calculated such that a set would get <0.5
    # annotations of a class, then that set gets 0 of that class.
    STRATIFIED = 'stratified'


def split_labels(
    labels_in: ImageLabels,
    # Reference set's max ratio (also capped by
    # TRAINING_BATCH_LABEL_COUNT) and validation set's ratio.
    # Remaining annotations go to the training set.
    # Example: (0.5, 0.15) for a 5% ref / 15% val / 80% train split.
    split_ratios: tuple[float, float],
    class_sampling: ClassSamplingMethod,
) -> TrainingTaskLabels:
    """
    Split annotation data into training, reference, and validation sets.
    """

    # Determine train/ref/val annotation count goals.
    # Note that these aren't necessarily going to be the final counts for
    # training.
    # If class filtering occurs after this function call, then the counts
    # will change.
    ref_max_ratio, val_ratio = split_ratios
    ref_goal_size = min(
        round(labels_in.label_count * ref_max_ratio),
        config.TRAINING_BATCH_LABEL_COUNT)
    val_goal_size = round(labels_in.label_count * val_ratio)
    train_goal_size = (
        labels_in.label_count - ref_goal_size - val_goal_size)

    # Preempt set-size ValueErrors that would be raised from
    # train_test_split().
    # TrainingLabelErrors are more specific and thus should be nicer for
    # error handling.

    if class_sampling == ClassSamplingMethod.IGNORE:
        set_minimum_size = 1
        explanation = "Each set must be non-empty."
    else:
        # Stratified by class
        set_minimum_size = len(labels_in.classes_set)
        explanation = (
            f"Each set's size must not be less than the number of classes"
            f" ({set_minimum_size}) to work with train_test_split().")
    if ref_goal_size < set_minimum_size \
            or val_goal_size < set_minimum_size \
            or train_goal_size < set_minimum_size:
        raise TrainingLabelsError(
            f"Not enough annotations to populate train/ref/val sets."
            f" Split was calculated as"
            f" {train_goal_size}/{ref_goal_size}/{val_goal_size}."
            f" {explanation}"
        )

    # Add annotations to the three sets.

    train_data = defaultdict(list)
    ref_data = defaultdict(list)
    val_data = defaultdict(list)

    # Map integer indices to image keys.
    # This is just a list, and the 'lookup keys' are the list indices.
    #
    # We want this so we have smaller identifiers for each image
    # (int instead of an arbitrary string)
    # to use in the potentially very long, RAM-consuming lists below.
    image_lookup = labels_in.image_keys
    # Flat lists of annotation identifiers (image index +
    # annotation-in-image index) and labels.
    annotation_indices_flat = []
    labels_flat = []

    for image_index, image_key in enumerate(image_lookup):
        for annotation_index, annotation in enumerate(labels_in[image_key]):
            _row, _column, label = annotation
            annotation_indices_flat.append((image_index, annotation_index))
            labels_flat.append(label)

    # Leave the split to scikit-learn. It can only split a set into two,
    # so we first make it do a train+ref / val split, then a train / ref
    # split.
    # See ClassSamplingMethod for details on stratification.

    train_ref_indices, val_indices = train_test_split(
        annotation_indices_flat,
        test_size=val_goal_size, random_state=0, shuffle=True,
        stratify=(
            None if class_sampling == ClassSamplingMethod.IGNORE
            else labels_flat),
    )

    annotation_indices_flat_reverse_lookup = {
        annotation_index_pair: list_index
        for list_index, annotation_index_pair
        in enumerate(annotation_indices_flat)
    }
    train_ref_labels_flat = []
    for annotation_index_pair in train_ref_indices:
        index_into_all_annotations = \
            annotation_indices_flat_reverse_lookup[annotation_index_pair]
        train_ref_labels_flat.append(labels_flat[index_into_all_annotations])

    train_indices, ref_indices = train_test_split(
        train_ref_indices,
        test_size=ref_goal_size, random_state=0, shuffle=True,
        stratify=(
            None if class_sampling == ClassSamplingMethod.IGNORE
            else train_ref_labels_flat),
    )

    for set_indices, set_data in [
        (train_indices, train_data),
        (ref_indices, ref_data),
        (val_indices, val_data),
    ]:
        for image_index, annotation_index in set_indices:
            image_key = image_lookup[image_index]
            set_data[image_key].append(
                labels_in[image_key][annotation_index])

    return TrainingTaskLabels(
        train=ImageLabels(train_data),
        ref=ImageLabels(ref_data),
        val=ImageLabels(val_data),
    )


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
    accepted_classes: set[int] | None = None,
    # See split_labels().
    split_ratios: tuple[float, float] = (0.1, 0.1),
    # See split_labels().
    class_sampling: ClassSamplingMethod = ClassSamplingMethod.STRATIFIED,
) -> TrainingTaskLabels:
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
        labels = split_labels(
            labels_in,
            split_ratios=split_ratios,
            class_sampling=class_sampling,
        )

    # Identify classes common to both train and ref.
    train_classes = labels.train.classes_set
    ref_classes = labels.ref.classes_set
    train_ref_classes = train_classes.intersection(ref_classes)

    # Further filter the classes if this arg was specified.
    # TODO: Filter by accepted_classes earlier, before the train/ref/val
    #  split, because that can potentially alter the annotation count a
    #  lot. Keep the train+ref class-set filtering step here, because that
    #  filtering step shouldn't alter the count much, and requires knowing
    #  the train and ref sets in the first place.
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
