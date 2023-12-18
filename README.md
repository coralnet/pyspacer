# PySpacer

[![CI Status](https://github.com/coralnet/pyspacer/actions/workflows/python-app.yml/badge.svg)](https://github.com/coralnet/pyspacer/actions/workflows/python-app.yml)
[![PyPI version](https://badge.fury.io/py/pyspacer.svg)](https://badge.fury.io/py/pyspacer)

PySpacer (AKA spacer) provides utilities to extract features from random point 
locations in images and then train classifiers over those features.
It is used in the vision backend of `https://github.com/coralnet/coralnet`.

Spacer currently supports python >=3.8.

## Installation

The spacer repo can be installed in three ways.
* Pip install -- for integration with other Python projects.
* Local clone -- ideal for testing and development.
* From Dockerfile -- the only option that supports Caffe, which is used for the legacy feature-extractor.

### Config

Setting spacer config variables is only necessary when using certain features. If you don't need S3 storage, and you won't load extractors remotely, you can skip this section.

See `CONFIGURABLE_VARS` in `config.py` for a full list of available variables, and for an explanation of when each variable must be configured or not.

Spacer's config variables can be set in any of the following ways:

1. As environment variables; recommended if you `pip install` the package. Each variable name must be prefixed with `SPACER_`:
   - `export SPACER_AWS_ACCESS_KEY_ID='YOUR_AWS_KEY_ID'`
   - `export SPACER_AWS_SECRET_ACCESS_KEY='YOUR_AWS_SECRET_KEY'`
   - `export SPACER_AWS_REGION='us-west-2'`
   - `export SPACER_EXTRACTORS_CACHE_DIR='/your/cache'`
2. In a `secrets.json` file in the same directory as this README; recommended for Docker builds and local clones. Example `secrets.json` contents:
   ```json
   {
     "AWS_ACCESS_KEY_ID": "YOUR_AWS_KEY_ID",
     "AWS_SECRET_ACCESS_KEY": "YOUR_AWS_SECRET_KEY",
     "AWS_REGION": "us-west-2",
     "EXTRACTORS_CACHE_DIR": "/your/cache"
   }
   ```
3. As a Django setting; recommended for a Django project that uses spacer. Example code in a Django settings module:
   ```python
   SPACER = {
       'AWS_ACCESS_KEY_ID': 'YOUR_AWS_KEY_ID',
       'AWS_SECRET_ACCESS_KEY': 'YOUR_AWS_SECRET_KEY',
       'AWS_REGION': 'us-west-2',
       'EXTRACTORS_CACHE_DIR': '/your/cache',
   }
   ```

Spacer supports the following schemes of using multiple settings sources:

- Check environment variables first, then use secrets.json as a fallback.
- Check environment variables first, then use Django settings as a fallback.

However, spacer will not read from multiple file-based settings sources; so if a secrets.json file is present, then spacer will not check for Django settings as a fallback.

To debug your configuration, try opening a Python shell and run `from spacer import config`, then `config.check()`.

### Docker build
The docker build is used in coralnet's deployment.
* Install docker on your system
* Clone this repo to a local folder; let's say it's `/your/local/pyspacer`
* Set up configuration as detailed above.
* Choose a local folder for caching extractor files; let's say it's `/your/local/cache`
* Build image: `docker build -f /your/local/pyspacer/Dockerfile -t myimagename`
* Run: `docker run -v /your/local/cache:/workspace/cache -v /your/local/pyspacer:/workspace/spacer -it myimagename`
  * The `-v /your/local/cache:/workspace/cache` part ensures 
that all build attempts use the same cache folder of your host storage.
  * The `-v /your/local/pyspacer:/workspace/spacer` mounts your local spacer clone (including 
`secrets.json`, if used) so that the container has the right permissions.
  * Overall, this runs the default CMD command specified in the dockerfile 
(unit-test with coverage). If you want to enter the docker container, 
run the same command but append `bash` in the end.

### Pip install
* `pip install pyspacer`
* Set up configuration. `secrets.json` isn't an option here, so either use environment variables, or Django settings if you have a Django project.

### Local clone
* Clone this repo
  * If using Windows: turn Git's `autocrlf` setting off before your initial checkout. Otherwise, pickled classifiers in `spacer/tests/fixtures` will get checked out with `\r\n` newlines, and the pickle module will fail to load them, leading to test failures. However, autocrlf should be left on when adding any new non-pickle files.
* `pip install -r requirements.txt`
* Set up configuration


## Code overview

Spacer executes tasks as defined in messages. The message types are defined
in `messages.py` and the tasks in `tasks.py`. Several data types which can be used for input and output serialization are defined
in `data_classes.py`.

For examples on how to create spacer tasks, refer to the Core API section below, and the unit tests in `test_tasks.py`.

Tasks can be executed directly by calling the methods in tasks.py. 
However, spacer also supports an interface with AWS Batch 
handled by `env_job()` in `mailman.py`. 

Spacer supports four storage types: `s3`, `filesystem`, `memory` and `url`.
 Refer to `storage.py` for details. The memory storage is mostly used for 
 testing, and the url storage is read only.

`config.py` defines configurable variables/settings and various constants.


## Feature extractors

The first step when analyzing an image, or preparing an image as training data, is extracting [features](https://en.wikipedia.org/wiki/Feature_(computer_vision)) from the image. Therefore, you need a feature extractor to use spacer, but spacer does not provide one out of the box.

Spacer's `extract_features.py` provides the Python classes `EfficientNetExtractor` for loading EfficientNet extractors in PyTorch format (CoralNet 1.0's default extraction scheme), and `VGG16CaffeExtractor` for loading VGG16 extractors in Caffe format (CoralNet's legacy extraction scheme).

You'll either want to match one of these schemes so you can use the provided classes, or you'll have to write your own extractor class which inherits from the base class `FeatureExtractor`. Between the provided classes, the easier one to use will probably be `EfficientNetExtractor`, because Caffe is old software which is more complicated to install.

If you're loading the extractor files remotely (from S3 or from a URL), the files will be automatically cached to your configured `EXTRACTORS_CACHE_DIR` for faster subsequent loads.


## Core API

The `tasks.py` module has four functions which comprise the main interface of pyspacer:

### extract_features

Takes an image, a list of pixel locations on that image, and a feature extractor. Produces a single feature vector out of the image-data at those pixel locations. Example:

```python
from spacer.extract_features import EfficientNetExtractor
from spacer.messages import DataLocation, ExtractFeaturesMsg
from spacer.tasks import extract_features

message = ExtractFeaturesMsg(
    # This token is purely for your bookkeeping; you may find it useful if you
    # choose to track tasks by saving these task messages. For example, you
    # can make the token something that uniquely identifies the input image.
    job_token='image1',
    # Instantiated feature extractor. Each extractor class defines the
    # data_locations which must be specified. In EfficientNetExtractor's case,
    # a PyTorch 'weights' file is required.
    extractor=EfficientNetExtractor(
        data_locations=dict(
            weights=DataLocation('filesystem', '/path/to/weights.pt'),
        ),
    ),
    # (row, column) tuples specifying pixel locations in the image.
    # Note that row is y, column is x.
    rowcols=[(2200, 1000), (1400, 1500), (3000, 450)],
    # Where the input image should be read from.
    image_loc=DataLocation('filesystem', '/path/to/image1.jpg'),
    # Where the feature vector should be output to.
    # CoralNet uses a custom .featurevector extension for these, but the
    # format is just JSON.
    feature_loc=DataLocation('filesystem', '/path/to/image1.featurevector'),
)
return_message = extract_features(message)
print("Feature vector stored at: /path/to/image1.featurevector")
print(f"Extraction runtime: {return_message.runtime:.1f} s")
```

### train_classifier

Takes:

- Feature vectors, each vector corresponding to a set of pixel locations in one image
- Ground-truth (typically human-confirmed) annotations corresponding to those feature vectors
- Optionally, previously-created classifiers to re-evaluate with these annotations
- Training parameters

Produces a classifier (model) loadable in scikit-learn, and classifier evaluation results. Example:

```python
from spacer.data_classes import ImageLabels
from spacer.messages import DataLocation, TrainClassifierMsg
from spacer.tasks import train_classifier
from spacer.task_utils import preprocess_labels

message = TrainClassifierMsg(
    # For your bookkeeping.
    job_token='classifier1',
    # 'minibatch' is currently the only trainer that spacer defines.
    trainer_name='minibatch',
    # How many iterations the training algorithm should run; more epochs
    # = more opportunity to converge to a better fit, but slower.
    nbr_epochs=10,
    # Classifier types available:
    # 1. 'MLP': multi-layer perceptron; newer classifier type for CoralNet
    # 2. 'LR': logistic regression; older classifier type for CoralNet
    clf_type='MLP',
    # Point-locations to ground-truth-labels (annotations) mappings
    # used to train the classifier.
    # The dict keys must be the same as the `key` used in the
    # extract-features task's `feature_loc`.
    # The dict values are lists of tuples of (row, column, label ID).
    # You'll need to be tracking a mapping of integer label IDs to the
    # labels you use.
    # preprocess_labels() can automatically split the data into training,
    # reference, and validation sets. However, you may also define how to
    # split it yourself; for details, see `TrainingTaskLabels` comments
    # in messages.py.
    labels=preprocess_labels(ImageLabels({
        '/path/to/image1.featurevector': [(1000, 2000, 1), (3000, 2000, 2)],
        '/path/to/image2.featurevector': [(1000, 2000, 3), (3000, 2000, 1)],
        '/path/to/image3.featurevector': [(1234, 2857, 11), (3094, 2262, 25)],
    })),
    # All the feature vectors should use the same storage_type, and the same
    # S3 bucket_name if applicable. This DataLocation's purpose is to describe
    # those common storage details. The key arg is ignored, because that will
    # be different for each feature vector.
    features_loc=DataLocation('filesystem', ''),
    # List of previously-created models (classifiers) to also evaluate
    # using this dataset, for informational purposes only.
    # A classifier is stored as a pickled CalibratedClassifierCV.
    previous_model_locs=[
        DataLocation('filesystem', '/path/to/oldclassifier1.pkl'),
        DataLocation('filesystem', '/path/to/oldclassifier2.pkl'),
    ],
    # Where the new model (classifier) should be output to.
    model_loc=DataLocation('filesystem', '/path/to/classifier1.pkl'),
    # Where the detailed evaluation results of the new model should be stored.
    valresult_loc=DataLocation('filesystem', '/path/to/valresult.json'),
)
return_message = train_classifier(message)
print("Classifier stored at: /path/to/classifier1.pkl")
print("Evaluation results stored at: /path/to/valresult.json")
print(f"New model's accuracy (0.0 = 0%, 1.0 = 100%): {return_message.acc}")
print(f"Previous models' accuracies: {return_message.pc_accs}")
print(
    "New model's accuracy progression (calculated on the reference set)"
    f" after each epoch of training: {return_message.ref_accs}")
print(f"Training runtime: {return_message.runtime:.1f} s")
```

Evaluation results consist of three arrays:

- `gt`: Ground-truth label IDs for each point in the validation set.
- `est`: Estimated (classifier-predicted) label IDs for each point.
- `scores`: Classifier's confidence scores (0.0 = 0%, 1.0 = 100%) for each estimated label ID.

The *i*th element of `gt`, *i*th element of `est`, and *i*th element of `scores` correspond to each other. But the elements are otherwise in an undefined order.

Accuracy is defined as the percentage of `gt` labels that match the corresponding `est` labels.

### classify_features

Takes a feature vector (representing points in an image) to classify, and a classifier trained on the same type of features (EfficientNet or VGG16). Produces prediction results (scores) for the image points, as posterior probabilities for each class. Example:

```python
from spacer.messages import DataLocation, ClassifyFeaturesMsg
from spacer.tasks import classify_features

message = ClassifyFeaturesMsg(
    # For your bookkeeping.
    job_token='image1',
    # Where the input feature-vector should be read from.
    feature_loc=DataLocation('filesystem', '/path/to/image1.featurevector'),
    # Where the classifier should be read from.
    classifier_loc=DataLocation('filesystem', '/path/to/classifier1.pkl'),
)
return_message = classify_features(message)
print(f"Classification runtime: {return_message.runtime:.1f} s")
print(f"Classes (recognized labels): {return_message.classes}")
print(
    "Classifier's scores for each point in the feature vector;"
    " scores are posterior probabilities of each class, with classes"
    " ordered as above:")
for row, col, scores in return_message.scores:
    print(f"Row {row}, column {col}: {scores}")
```

The label which has the highest score for a particular point (row-column position) can be considered the classifier's predicted label for that point.

One possible usage strategy is to trust the classifier's predictions for points where the highest confidence score is above a certain threshold, such as 0.8 (80%), and have human annotators check all other points.

### classify_image

This basically does `extract_features` and `classify_features` together in one go, without needing to specify a storage location for the feature vector.

Takes an image, a list of pixel locations on that image, a feature extractor, and a classifier. Produces prediction results (scores) for the image points, as posterior probabilities for each class. Example:

```python
from spacer.extract_features import EfficientNetExtractor
from spacer.messages import DataLocation, ClassifyImageMsg
from spacer.tasks import classify_image

message = ClassifyImageMsg(
    # For your bookkeeping.
    job_token='image1',
    # Where the input image should be read from.
    image_loc=DataLocation('filesystem', '/path/to/image1.jpg'),
    # Instantiated feature extractor.
    extractor=EfficientNetExtractor(
        data_locations=dict(
            weights=DataLocation('filesystem', '/path/to/weights.pt'),
        ),
    ),
    # (row, column) tuples specifying pixel locations in the image.
    # Note that row is y, column is x.
    rowcols=[(2200, 1000), (1400, 1500), (3000, 450)],
    # Where the classifier should be read from.
    classifier_loc=DataLocation('filesystem', '/path/to/classifier1.pkl'),
)
return_message = classify_image(message)
print(f"Runtime: {return_message.runtime:.1f} s")
print(f"Classes (recognized labels): {return_message.classes}")
print(
    "Classifier's scores for each point in rowcols;"
    " scores are posterior probabilities of each class, with classes"
    " ordered as above:")
for row, col, scores in return_message.scores:
    print(f"Row {row}, column {col}: {scores}")
```


## Unit tests

Run the test suite by running `python -m unittest` from the `spacer` directory.

- Expect many tests to be skipped, since most test fixtures aren't set up for public access yet.

- Run just a single test module with a command like `python -m unittest tests.test_tasks`, or just `python -m tests.test_tasks` (the latter invokes the `if __name__ == '__main__':` part of the module).

- You can get logging output during test runs with the `LOG_DESTINATION` and `LOG_LEVEL` vars in config.py.

If you are using the docker build or local install, 
you can check code coverage like so:
```
    coverage run --source=spacer --omit=spacer/tests/* -m unittest    
    coverage report -m
    coverage html
```
