# PySpacer

[![Build Status](https://travis-ci.com/beijbom/pyspacer.svg?branch=master)](https://travis-ci.com/beijbom/pyspacer)
[![PyPI version](https://badge.fury.io/py/pyspacer.svg)](https://badge.fury.io/py/pyspacer)

This repository provide utilities to extract features from random point 
locations in images and then training classifiers over those features.
It is used in the vision backend of `https://github.com/beijbom/coralnet`.

Spacer currently supports python >=3.5.

### Overview
Spacer executes tasks as defined in messages. The messages types are defined
in `messages.py` and the tasks in `tasks.py`. We also define several data-types
in `data_classes.py` which define input and output types. 

Refer to the unit-test in `test_tasks.py` for examples on how to create tasks.

Tasks can be executed directly by calling the methods in tasks.py. 
However, spacer also supports an interface with SQS 
handled by `sqs_fetch()` in `mailman.py`. 

Spacer supports four storage types: `s3`, `filesystem`, `memory` and `url`.
 Refer to `storage.py` for details. The Memory storage is mostly used for 
 testing, and the `url` storage is read only.

Also take a look at `config.py` for settings and configuration. 

### Installation

The spacer repo can be installed in three ways.
* Using Docker -- the only option that supports Caffe.
* Local clone -- ideal for testing and development.
* Using pip install -- for integration in other code-bases.

#### Config
Spacer needs three variables. They can either be set
as environmental variables (recommended if you `pip install` the package), 
or in a `secrets.json` file in the same directory as this README 
(recommended for Docker builds and local clones). 
The `secrets.json` should look like this.
```json
{
  "SPACER_AWS_ACCESS_KEY_ID": "YOUR_AWS_KEY_ID",
  "SPACER_AWS_SECRET_ACCESS_KEY": "YOUR_AWS_SECRET_KEY",
  "SPACER_LOCAL_MODEL_PATH": "/path/to/your/local/models"
}
``` 

#### Docker build
The docker build is the preferred build and the one used in deployment.
* Install docker on your system
* Create `secrets.json` as detailed above.
* Create folder `/path/to/your/local/models` for caching model files.
* Build image: `docker build -t spacer:test .`
* Run: `docker run -v /path/to/your/local/models:/workspace/models -v ${PWD}:/workspace/spacer/ -it spacer:test`

The `-v /path/to/your/local/models:/workspace/models` part will make sure 
the downloaded models are cached to your host storage. 
which makes rerunning stuff much faster.

The `-v ${PWD}:/workspace/spacer/` mounts your current folder including 
`secrets.json` so that the container has the right permissions.

The last step will run the default CMD command specified in the dockerfile 
(unit-test with coverage). If you want to enter the docker container 
run the same command but append `bash` in the end: 

```
docker run -v /path/to/your/local/models:/workspace/models -v ${PWD}:/workspace/spacer/ -it spacer:test
```

#### Pip install
* `pip install spacer`
* Set environmental variables.

#### Local clone
* Clone this repo
* `pip install -r requirements.txt`

### Code coverage
If you are using the docker build or local install, 
you can check code coverage like so:
```
    coverage run --source=spacer --omit=spacer/tests/* -m unittest    
    coverage report -m
    coverage html
```
