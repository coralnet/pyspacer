from __future__ import annotations
from collections import defaultdict
from enum import Enum
from logging import getLogger
import random

from PIL import Image
from sklearn.model_selection import train_test_split

from spacer import config
from spacer.data_classes import ImageLabels, LabelId
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


class SplitMode(Enum):
    """
    How to split annotations between train, ref, and val sets.
    """

    # Each feature vector's points all go into train, or all go into ref, or
    # all go into val.
    #
    # This mode is based on a few assumptions:
    # 1) Each feature vector to be used as training data represents one image.
    # 2) The imagery around a particular point is more similar to the imagery
    #    around another point if those two points are from the same image, as
    #    opposed to being from different images.
    #    Therefore, a classifier trained on points of image A is expected to
    #    have better-or-equal accuracy on other points of image A, compared to
    #    its accuracy on points of image B.
    # 3) Real-world usage of classifiers involves classifying 'new' images
    #    which the classifier has not been trained on.
    #
    # Thus, this mode can help ensure real-world applicability of
    # classifiers, in terms of usefulness of calibration (which depends on the
    # differences between train and ref), and rigor of the evaluation results
    # (which depends on the differences between train and val).
    #
    # However, the annotation count may not end up precisely balanced
    # between train/ref/val as desired, particularly when the feature vector
    # size is comparable to the set size. For example, if each feature vector
    # has 100 points, and the target ref-set size is 450, then the best we can
    # do is giving the ref set either 400 or 500 points.
    VECTORS = 'vectors'

    # The split is done on an individual point basis, so a single
    # feature vector may be split across train/ref/val. For example, a
    # feature vector with 20 point-features may have 16 points going to train,
    # 2 going to ref, and 2 going to val.
    #
    # This allows the annotation count to be more precisely balanced
    # between train/ref/val.
    # However, there may be concerns that the imagery going into each set is
    # too similar, particularly when points are densely distributed within
    # each image.
    POINTS = 'points'

    # Stratified sampling by class: an A%/B%/C% train/ref/val split means
    # an A%/B%/C% split of each class.
    # The split is done on an individual point basis.
    #
    # The POINTS mode's results should already be approximately stratified due
    # to the annotations being shuffled. However, POINTS_STRATIFIED makes the
    # stratification more guaranteed. This can be useful because it makes the
    # final number of unique classes more consistent.
    #
    # Stratification checks that the number of annotations in each
    # set isn't less than the number of unique classes.
    # However, each set is NOT guaranteed to have at least 1 of each class.
    # If stratification is calculated such that a set would get <0.5
    # annotations of a class, then that set gets 0 of that class.
    POINTS_STRATIFIED = 'points_stratified'


def split_labels(
    labels_in: ImageLabels,
    # Reference set's max ratio (also capped by
    # TRAINING_BATCH_LABEL_COUNT) and validation set's ratio.
    # Remaining annotations go to the training set.
    # Example: (0.05, 0.15) for a 5% ref / 15% val / 80% train split.
    split_ratios: tuple[float, float],
    split_mode: SplitMode,
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

    if split_mode == SplitMode.POINTS_STRATIFIED:
        # Stratified by class
        set_minimum_size = len(labels_in.classes_set)
        explanation = (
            f"Each set's size must not be less than the number of classes"
            f" ({set_minimum_size}) to work with train_test_split().")
    else:
        set_minimum_size = 1
        explanation = "Each set must be non-empty."
    if ref_goal_size < set_minimum_size \
            or val_goal_size < set_minimum_size \
            or train_goal_size < set_minimum_size:
        raise TrainingLabelsError(
            f"Not enough annotations to populate train/ref/val sets."
            f" Split was calculated as"
            f" {train_goal_size}/{ref_goal_size}/{val_goal_size}."
            f" {explanation}"
        )

    if split_mode == SplitMode.VECTORS:

        # We don't use train_test_split() in this case because:
        # 1) train_test_split() doesn't seem to be able to split annotations
        #    accurately when splitting must be done at the feature vector
        #    level, since each vector may have different annotation counts.
        # 2) the main benefit of train_test_split() is stratification, but
        #    we're not doing that in this case anyway.

        image_keys = list(labels_in.keys())
        random.seed(0)
        random.shuffle(image_keys)
        # Use a generator so we can continue iterating over the image keys
        # across multiple loops.
        image_key_generator = (key for key in image_keys)

        train_labels = dict()
        ref_labels = dict()
        val_labels = dict()

        for labels, goal_size in [
            (val_labels, val_goal_size),
            (ref_labels, ref_goal_size),
            (train_labels, train_goal_size),
        ]:
            # Add annotations to the set until the goal size is met/exceeded,
            # or until all annotations are added.
            # If the goal's exceeded, no annotations are taken back. So, since
            # the order the sets are populated is val-ref-train, val and ref
            # will either meet or exceed the goal sizes, while train will
            # either meet or fall short of the goal size.
            num_annotations = 0
            while num_annotations < goal_size:
                try:
                    key = next(image_key_generator)
                except StopIteration:
                    break
                labels[key] = labels_in[key]
                num_annotations += len(labels_in[key])

        return TrainingTaskLabels(
            train=ImageLabels(train_labels),
            ref=ImageLabels(ref_labels),
            val=ImageLabels(val_labels),
        )

    # From here on out, we have either SplitMode.POINTS or
    # SplitMode.POINTS_STRATIFIED, meaning we'll add annotations to the three
    # sets on an individual-point basis.

    train_data = defaultdict(list)
    ref_data = defaultdict(list)
    val_data = defaultdict(list)

    # Map integer indices to image keys.
    # This is just a list, and the 'lookup keys' are the list indices.
    #
    # We want this so we have smaller identifiers for each image
    # (int instead of an arbitrary string)
    # to use in the potentially very long, RAM-consuming lists below.
    image_lookup = list(labels_in.keys())
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
    # See SplitMode for details on stratification.

    train_ref_indices, val_indices = train_test_split(
        annotation_indices_flat,
        test_size=val_goal_size, random_state=0, shuffle=True,
        stratify=(
            labels_flat if split_mode == SplitMode.POINTS_STRATIFIED
            else None),
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
            train_ref_labels_flat if split_mode == SplitMode.POINTS_STRATIFIED
            else None),
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
    accepted_classes: set[LabelId] | None = None,
    # See split_labels().
    split_ratios: tuple[float, float] = (0.1, 0.1),
    # See split_labels().
    split_mode: SplitMode = SplitMode.VECTORS,
) -> TrainingTaskLabels:
    """
    This function can be used to preprocess labels before creating a
    TrainClassifierMsg. It does:
    1) Splitting of labels into train/ref/val, if not already done
    2) Filtering of labels by accepted_classes, if that arg is specified
    3) Error checks

    This is also called by pyspacer after receiving a TrainClassifierMsg
    to ensure the error checks are done.
    """

    if isinstance(labels_in, TrainingTaskLabels):

        # The caller has decided how to split the data into
        # training, reference, and validation sets.

        labels = labels_in

        if accepted_classes:
            for set_name in ['train', 'ref', 'val']:
                labels[set_name] = \
                    labels[set_name].filter_classes(accepted_classes)

    else:

        # The caller is leaving the split to pyspacer.

        pre_split_labels = labels_in

        if accepted_classes:
            pre_split_labels = pre_split_labels.filter_classes(
                accepted_classes)

        if split_mode == SplitMode.POINTS_STRATIFIED:

            # train_test_split() will want each class to have at least as many
            # annotations as sets (even though it doesn't guarantee what each
            # set ends up with).
            # Otherwise it'll raise an error. So, filter out the rare classes
            # first.
            common_enough_classes = [
                label
                for label, count in labels_in.label_count_per_class.items()
                if count >= 3]
            if len(common_enough_classes) != len(labels_in.classes_set):
                pre_split_labels = pre_split_labels.filter_classes(
                    common_enough_classes)

        labels = split_labels(
            pre_split_labels,
            split_ratios=split_ratios,
            split_mode=split_mode,
        )

    # Identify classes common to both train and ref.
    train_classes = labels.train.classes_set
    ref_classes = labels.ref.classes_set
    train_ref_classes = train_classes.intersection(ref_classes)

    if len(train_ref_classes) <= 1:
        raise TrainingLabelsError(
            f"Need multiple classes to do training."
            f" After preprocessing training data, there are"
            f" {len(train_ref_classes)} class(es) left.")

    for set_name in ['train', 'ref', 'val']:
        if not labels[set_name].classes_set.issubset(train_ref_classes):
            # The classifier can only learn about classes which are in
            # both train and ref, so if a class is missing from one (or both)
            # of train or ref, we discard that class's annotations.
            # Note that this may change the annotation-count balance between
            # train / ref / val.
            labels[set_name] = \
                labels[set_name].filter_classes(train_ref_classes)

        if len(labels[set_name]) == 0:
            raise TrainingLabelsError(
                f"After preprocessing training data,"
                f" '{set_name}' set is empty.")

    return labels
