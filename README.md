# PySpacer

[![CI Status](https://github.com/coralnet/pyspacer/actions/workflows/python-app.yml/badge.svg)](https://github.com/coralnet/pyspacer/actions/workflows/python-app.yml)
[![PyPI version](https://badge.fury.io/py/pyspacer.svg)](https://badge.fury.io/py/pyspacer)

This repository provides utilities to extract features from random point 
locations in images and then train classifiers over those features.
It is used in the vision backend of `https://github.com/coralnet/coralnet`.

Spacer currently supports python >=3.8.

## Installation

The spacer repo can be installed in three ways.
* Pip install -- for integration with other Python projects.
* Local clone -- ideal for testing and development.
* From Dockerfile -- the only option that supports Caffe, which is used for the legacy feature-extractor.

### Config
Spacer's config variables can be set in any of the following ways:

1. As environment variables; recommended if you `pip install` the package. Each variable name must be prefixed with `SPACER_`:
   - `export SPACER_AWS_ACCESS_KEY_ID='YOUR_AWS_KEY_ID'`
   - `export SPACER_AWS_SECRET_ACCESS_KEY='YOUR_AWS_SECRET_KEY'`
   - `export SPACER_LOCAL_MODEL_PATH='/path/to/your/local/models'`
2. As a Django setting; recommended for a Django project that uses spacer. Example code in a Django settings module:
   ```python
   SPACER = {
       'AWS_ACCESS_KEY_ID': 'YOUR_AWS_KEY_ID',
       'AWS_SECRET_ACCESS_KEY': 'YOUR_AWS_SECRET_KEY',
       'LOCAL_MODEL_PATH': '/path/to/your/local/models',
   }
   ```
3. In a `secrets.json` file in the same directory as this README; recommended for Docker builds and local clones. Example `secrets.json` contents:
   ```json
   {
     "AWS_ACCESS_KEY_ID": "YOUR_AWS_KEY_ID",
     "AWS_SECRET_ACCESS_KEY": "YOUR_AWS_SECRET_KEY",
     "LOCAL_MODEL_PATH": "/path/to/your/local/models"
   }
   ```
   
LOCAL_MODEL_PATH is required. The two AWS access variables are required unless spacer is running on an AWS instance which has been set up with `aws configure`. The rest of the config variables are optional; see `CONFIGURABLE_VARS` in `config.py` for a full list.

To debug your configuration, try opening a Python shell and run `from spacer import config`, then `config.check()`.

### Docker build
The docker build is used in coralnet's deployment.
* Install docker on your system
* Set up configuration as detailed above.
* Create folder `/path/to/your/local/models` for caching model files.
* Build image: `docker build -t spacer:test .`
* Run: `docker run -v /path/to/your/local/models:/workspace/models -v ${PWD}:/workspace/spacer/ -it spacer:test`

The `-v /path/to/your/local/models:/workspace/models` part will make sure 
the downloaded models are cached to your host storage. 
which makes rerunning stuff much faster.

The `-v ${PWD}:/workspace/spacer/` mounts your current folder (including 
`secrets.json`, if used) so that the container has the right permissions.

The last step will run the default CMD command specified in the dockerfile 
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

Refer to the unit tests in `test_tasks.py` for examples on how to create tasks.

Tasks can be executed directly by calling the methods in tasks.py. 
However, spacer also supports an interface with AWS Batch 
handled by `env_job()` in `mailman.py`. 

Spacer supports four storage types: `s3`, `filesystem`, `memory` and `url`.
 Refer to `storage.py` for details. The memory storage is mostly used for 
 testing, and the url storage is read only.

`config.py` defines configurable variables/settings and various constants.


## Core API

The `tasks.py` module has four functions which comprise the main interface of pyspacer:

### extract_features

Takes an image, and a list of pixel locations on that image. Produces a single feature vector out of the image-data at those pixel locations. Example:

```python
from spacer.messages import DataLocation, ExtractFeaturesMsg
from spacer.tasks import extract_features

message = ExtractFeaturesMsg(
    # Identifier to set this extraction job apart from others. Makes sense
    # to use something that uniquely identifies the image.
    job_token='image_123',
    # Extractors available:
    # 1. 'efficientnet_b0_ver1': Generally recommended
    # 2. 'vgg16_coralnet_ver1': Legacy, requires Caffe
    # 3. 'dummy': Produces feature vectors which are in the correct format but
    # don't have meaningful data. The fastest extractor; can help for testing.
    feature_extractor_name='efficientnet_b0_ver1',
    # (row, column) tuples specifying pixel locations in the image.
    # Note that row is y, column is x.
    rowcols=[(2200, 1000), (1400, 1500), (3000, 450)],
    # Where the input image should be read from.
    image_loc=DataLocation(
        storage_type='filesystem',
        key='/path/to/image',
    ),
    # Where the feature vector should be output to.
    feature_loc=DataLocation(
        storage_type='filesystem',
        key='/path/to/feature/vector',
    ),
)
return_message = extract_features(message)
print("Feature vector stored at:")
print(f"Runtime: {return_message.runtime}")
```

### train_classifier

TODO

### classify_features

TODO

### classify_image

TODO


## Code coverage

If you are using the docker build or local install, 
you can check code coverage like so:
```
    coverage run --source=spacer --omit=spacer/tests/* -m unittest    
    coverage report -m
    coverage html
```
