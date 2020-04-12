"""
Defines the highest level methods for completing tasks.
"""
import time

from spacer.data_classes import ImageLabels, ImageFeatures
from spacer.extract_features import feature_extractor_factory
from spacer.messages import \
    ExtractFeaturesMsg, \
    ExtractFeaturesReturnMsg, \
    TrainClassifierMsg, \
    TrainClassifierReturnMsg, \
    ClassifyFeaturesMsg, \
    ClassifyImageMsg, \
    ClassifyReturnMsg
from spacer.storage import load_image, load_classifier, store_classifier
from spacer.task_utils import check_rowcols
from spacer.train_classifier import trainer_factory


def extract_features(msg: ExtractFeaturesMsg) -> ExtractFeaturesReturnMsg:

    print("-> Extracting features for job:{}.".format(msg.job_token))
    extractor = feature_extractor_factory(msg.feature_extractor_name)
    img = load_image(msg.image_loc)
    check_rowcols(msg.rowcols, img)
    features, return_msg = extractor(img, msg.rowcols)
    features.store(msg.feature_loc)
    return return_msg


def train_classifier(msg: TrainClassifierMsg) -> TrainClassifierReturnMsg:

    print("Training classifier pk:{}.".format(msg.job_token))
    trainer = trainer_factory(msg.trainer_name)

    # Do the actual training
    clf, val_results, return_message = trainer(
        ImageLabels.load(msg.traindata_loc),
        ImageLabels.load(msg.traindata_loc),
        msg.nbr_epochs,
        [load_classifier(loc) for loc in msg.previous_model_locs],
        msg.features_loc
    )

    # Store
    store_classifier(msg.model_loc, clf)
    val_results.store(msg.valresult_loc)

    return return_message


def classify_features(msg: ClassifyFeaturesMsg) -> ClassifyReturnMsg:

    t0 = time.time()
    features = ImageFeatures.load(msg.feature_loc)

    clf = load_classifier(msg.classifier_loc)

    scores = [(pf.row, pf.col, clf.predict_proba(pf.data_np).tolist()[0]) for
              pf in features.point_features]

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
    scores = [(row, col, clf.predict_proba(features.get_array((row, col))).
               tolist()) for row, col in msg.rowcols]

    # Return
    return ClassifyReturnMsg(
        runtime=time.time() - t0,
        scores=scores,
        classes=list(clf.classes_),
        valid_rowcol=True)
