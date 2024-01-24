"""
Defines the highest level methods for completing tasks.
"""
import time
import traceback
from typing import Callable
from spacer import config
from spacer.data_classes import ImageFeatures
from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    ClassifyFeaturesMsg, \
    ClassifyImageMsg, \
    ClassifyReturnMsg, JobMsg, JobReturnMsg
from spacer.storage import load_image, load_classifier, store_classifier
from spacer.task_utils import check_extract_inputs, preprocess_labels
from spacer.train_classifier import ClassifierTrainer, trainer_factory

logger = getLogger(__name__)


def extract_features(msg: ExtractFeaturesMsg) -> ExtractFeaturesReturnMsg:

    img = load_image(msg.image_loc)

    check_extract_inputs(img, msg.rowcols, msg.image_loc.key)

    check_rowcols(msg.rowcols, img)
    if not isinstance(msg.extractor, Callable):
        raise TypeError("msg.extractor must be callable")
    
    with config.log_entry_and_exit('actual extraction'):
        features, return_msg = msg.extractor(img, msg.rowcols)
    if not hasattr(features, 'store'):
        raise TypeError("features object must have a store method")

    with config.log_entry_and_exit('storing features'):
        features.store(msg.feature_loc)

    return return_msg


def train_classifier(msg: TrainClassifierMsg) -> TrainClassifierReturnMsg:
    trainer: ClassifierTrainer = trainer_factory(msg.trainer_name)

    labels = preprocess_labels(msg.labels)
    logger.debug(
        f"Unique classes:"
        f" Train + Ref = {len(labels.ref.classes_set)},"
        f" Val = {len(labels.val.classes_set)}")
    logger.debug(
        f"Label count:"
        f" Train = {labels.train.label_count},"
        f" Ref = {labels.ref.label_count},"
        f" Val = {labels.val.label_count},"
        f" Total = {labels.label_count}")

    # Do the actual training
    with config.log_entry_and_exit('actual training'):
        clf, val_results, return_message = trainer(
            labels,
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
    check_extract_inputs(img, msg.rowcols, msg.image_loc.key)

    # Extract features
    features, _ = msg.extractor(img, msg.rowcols)

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
            logger.error('Error executing job {}: {}'.format(
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
