"""
Defines the highest level methods for completing tasks.
"""
import logging
import time
import traceback

from spacer import config
from spacer.data_classes import ImageFeatures
from spacer.extract_features import feature_extractor_factory
from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    ClassifyFeaturesMsg, \
    ClassifyImageMsg, \
    ClassifyReturnMsg, JobMsg, JobReturnMsg
from spacer.storage import load_image, load_classifier, store_classifier
from spacer.task_utils import check_rowcols
from spacer.train_classifier import trainer_factory


def extract_features(msg: ExtractFeaturesMsg) -> ExtractFeaturesReturnMsg:

    extractor = feature_extractor_factory(msg.feature_extractor_name)
    img = load_image(msg.image_loc)

    assert img.width * img.height <= config.MAX_IMAGE_PIXELS, \
        "Image ({}, {}) with {} pixels too large. (max: {})".format(
            img.width, img.height, img.width * img.height,
            config.MAX_IMAGE_PIXELS)
    assert len(msg.rowcols) <= config.MAX_POINTS_PER_IMAGE, \
        "Too many rowcol locations ({}). Max {} allowed".format(
            len(msg.rowcols), config.MAX_POINTS_PER_IMAGE
        )

    check_rowcols(msg.rowcols, img)

    with config.log_entry_and_exit('actual extraction'):
        features, return_msg = extractor(img, msg.rowcols)

    with config.log_entry_and_exit('storing features'):
        features.store(msg.feature_loc)

    return return_msg


def train_classifier(msg: TrainClassifierMsg) -> TrainClassifierReturnMsg:
    trainer = trainer_factory(msg.trainer_name)

    # Do the actual training
    with config.log_entry_and_exit('actual training'):
        clf, val_results, return_message = trainer(
            msg.train_labels,
            msg.val_labels,
            msg.nbr_epochs,
            [load_classifier(loc) for loc in msg.previous_model_locs],
            msg.features_loc,
            msg.clf_type
        )

    with config.log_entry_and_exit('storing classifier and val res'):
        store_classifier(msg.model_loc, clf)
        val_results.store(msg.valresult_loc)

    return return_message


def classify_features(msg: ClassifyFeaturesMsg) -> ClassifyReturnMsg:

    t0 = time.time()
    features = ImageFeatures.load(msg.feature_loc)

    clf = load_classifier(msg.classifier_loc)

    scores = [(pf.row, pf.col,
               clf.predict_proba(pf.data.reshape(1, -1)).tolist()[0])
              for pf in features.point_features]

    # Return
    return ClassifyReturnMsg(
        runtime=time.time() - t0,
        scores=scores,
        classes=clf.classes_.tolist(),
        valid_rowcol=features.valid_rowcol)


def classify_image(msg: ClassifyImageMsg) -> ClassifyReturnMsg:

    t0 = time.time()

    # Download image
    img = load_image(msg.image_loc)
    check_rowcols(msg.rowcols, img)

    # Extract features
    extractor = feature_extractor_factory(msg.feature_extractor_name)
    features, _ = extractor(img, msg.rowcols)

    # Classify
    clf = load_classifier(msg.classifier_loc)
    scores = [(row, col,
               clf.predict_proba(features.get_array((row, col))).tolist()[0])
              for row, col in msg.rowcols]

    # Return
    return ClassifyReturnMsg(
        runtime=time.time() - t0,
        scores=scores,
        classes=clf.classes_.tolist(),
        valid_rowcol=True)


def process_job(job_msg: JobMsg) -> JobReturnMsg:

    run = {
        'extract_features': extract_features,
        'train_classifier': train_classifier,
        'classify_features': classify_features,
        'classify_image': classify_image,
    }

    assert isinstance(job_msg, JobMsg)
    assert job_msg.task_name in config.TASKS

    results = []
    for task in job_msg.tasks:
        try:
            with config.log_entry_and_exit('{} [{}]'.format(
                    job_msg.task_name, task.job_token)):
                results.append(run[job_msg.task_name](task))
        except Exception:
            logging.error('Error executing job {}: {}'.format(
                task.job_token, traceback.format_exc()))
            return_msg = JobReturnMsg(
                original_job=job_msg,
                ok=False,
                results=None,
                error_message=traceback.format_exc()
            )
            return return_msg

    return_msg = JobReturnMsg(
        original_job=job_msg,
        ok=True,
        results=results,
        error_message=None
    )

    return return_msg
