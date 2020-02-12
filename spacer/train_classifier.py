import random
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier


def do_training(traindict, nbr_epochs, bucket):
    def get_unique_classes(keylist):
        labels = set()
        for imkey in keylist:
            labels = labels.union(set(traindict[imkey]))
        return labels

    # Calculate max nbr images to keep in memory (based on 5000 samples total).
    samples_per_image = len(traindict[traindict.keys()[0]])
    max_imgs_in_memory = 5000 / samples_per_image

    # Make train and ref split.
    # Reference set is here a hold-out part of the train-data portion.
    # Purpose of refset is to
    # 1) know accuracy per epoch
    # 2) calibrate classifier output scores.
    # We call it 'ref' to disambiguate from the validation set.
    imkeys = traindict.keys()
    refset = imkeys[::10]
    random.shuffle(refset)
    refset = refset[
             :max_imgs_in_memory]  # Make sure we don't go over the memory limit.
    trainset = list(set(imkeys) - set(refset))
    print("trainset: {}, valset: {} images".format(len(trainset), len(refset)))

    # Figure out # images per mini-batch and batches per epoch.
    images_per_minibatch = min(max_imgs_in_memory, len(trainset))
    n = int(np.ceil(len(trainset) / float(images_per_minibatch)))
    print(
        "Using {} images per mini-batch and {} mini-batches per epoch".format(
            images_per_minibatch, n))

    # Identify classes common to both train and val.
    # This will be our labelset for the training.
    trainclasses = get_unique_classes(trainset)
    refclasses = get_unique_classes(refset)
    classes = list(trainclasses.intersection(refclasses))
    print(
        "trainset: {}, valset: {}, common: {} labels".format(len(trainclasses),
                                                             len(refclasses),
                                                             len(classes)))
    if len(classes) == 1:
        raise ValueError('Not enough classes to do training (only 1)')

    # Load reference data (must hold in memory for the calibration)
    print("Loading reference data.")
    refx, refy = _load_mini_batch(traindict, refset, classes, bucket)

    # Initialize classifier and ref set accuracy list
    print("Online training...")
    clf = SGDClassifier(loss='log', average=True)
    refacc = []
    for epoch in range(nbr_epochs):
        print("Epoch {}".format(epoch))
        random.shuffle(trainset)
        mini_batches = _chunkify(trainset, n)
        for mb in mini_batches:
            x, y = _load_mini_batch(traindict, mb, classes, bucket)
            clf.partial_fit(x, y, classes=classes)
        refacc.append(_acc(refy, clf.predict(refx)))
        print("acc: {}".format(refacc[-1]))

    print("Calibrating.")
    clf_calibrated = CalibratedClassifierCV(clf, cv="prefit")
    clf_calibrated.fit(refx, refy)

    return clf_calibrated, refacc


def _evaluate_classifier(clf, imkeys, gtdict, classes, bucket):
    """
    Return the accuracy of classifier "clf" evaluated on "imkeys"
    with ground truth given in "gtdict". Features are fetched from S3 "bucket".
    """
    scores, gt, est = [], [], []
    for imkey in imkeys:
        x, y = _load_data(gtdict, imkey, classes, bucket)
        if len(x) > 0:
            scores.extend(list(clf.predict_proba(x)))
            est.extend(clf.predict(x))
            gt.extend(y)

    maxscores = [np.max(score) for score in scores]

    return gt, est, maxscores


def _chunkify(lst, n):
    return [lst[i::n] for i in xrange(n)]


def _load_data(labeldict, imkey, classes, bucket):
    """
    load feats from S3 and returns them along with the ground truth labels
    """
    key = bucket.get_key(imkey)

    # Load features
    x = json.loads(key.get_contents_as_string())

    # Load labels
    y = labeldict[imkey]

    # Remove samples for which the label is not in classes.
    if set(classes).isdisjoint(y):
        # none of the points for this image is in the labelset.
        return [], []
    else:
        x, y = zip(*[(xmember, ymember) for xmember, ymember in zip(x, y) if
                     ymember in classes])
        return x, y


def _load_mini_batch(labeldict, imkeylist, classes, bucket):
    """
    An interator over _load_data.
    """
    x, y = [], []
    for imkey in imkeylist:
        thisx, thisy = _load_data(labeldict, imkey, classes, bucket)
        x.extend(thisx)
        y.extend(thisy)
    return x, y


def _download_nets(name):
    conn = boto.connect_s3()
    bucket = conn.get_bucket('spacer-tools')
    was_cashed = False
    for suffix in ['.deploy.prototxt', '.caffemodel']:
        was_cashed = _download_file(bucket, name + suffix,
                                    os.path.join(config.LOCAL_MODEL_PATH,
                                                 name + suffix))
    return was_cashed


def _download_file(bucket, keystring, destination):
    if not os.path.isfile(destination):
        print("downloading {}".format(keystring))
        key = Key(bucket, keystring)
        key.get_contents_to_filename(destination)
        return False
    else:
        return True


def _acc(gt, est):
    """
    Calculate the accuracy of (agreement between) two interger valued list.
    """
    if len(gt) == 0 or len(est) == 0:
        raise TypeError('Inputs can not be empty')

    if not len(gt) == len(est):
        raise ValueError('Input gt and est must have the same length')

    for g in gt:
        if not isinstance(g, int):
            raise TypeError('Input gt must be an array of ints')

    for e in est:
        if not isinstance(e, int):
            raise TypeError('Input est must be an array of ints')

    return float(sum([(g == e) for (g, e) in zip(gt, est)])) / len(gt)
