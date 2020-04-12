# PySpacer

[![Build Status](https://travis-ci.org/beijbom/pyspacer.svg?branch=master)](https://travis-ci.org/beijbom/pyspacer)

This repository provide utilities to extract features from random point 
locations in images and then training classifiers over those features.
It is used in the vision backend of `https://github.com/beijbom/coralnet`.

Spacer currently supports python >=3.5.

### Overview
Spacer executes tasks as defined in messages. The messages types are defined
in `messages.py` and the tasks in `tasks.py`. We also define several data-types
in `data_classes.py` which define input and output types. 

Refer to the unit-test in `test_tasks.py` for examples on how to create tasks.
Currently the `extract_features` task only has a valid implementation 
through caffe, which requires the Docker build. We will add a PyTorch based
feature extractor soon.

Tasks can be executed directly by calling the methods in tasks.py. 
However, spacer also supports an interface with SQS 
handled by `sqs_mailman()` in `mailman.py`. 

Spacer supports there types of storage, s3, filesystem and memory. 
Refer to `storage.py` for details. The Memory storage is mostly for testing.

Also take a look at `config.py` for settings and configuration. 

### Installation

The spacer repo can be installed in three ways.
* Using Docker -- the only option that supports Caffe.
* Local clone -- ideal for fast testing and development.
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
* Run: `docker run -v /path/to/your/local/models:/workspace/models -it spacer:test`

The `-v /path/to/your/local/models:/workspace/models` part will make sure 
the downloaded models are cached to your local disk (outside the container), 
which makes rerunning stuff much faster.

The last step will run the default CMD command specified in the dockerfile 
(unit-test with coverage). If you want to enter the docker container 
run the same command but append `bash` in the end: 

```
docker run -v /path/to/your/local/models:/workspace/models -it test:Dockerfile bash
```

#### Pip install
* Install virtualenv.
* Set environmental variables.
* `pip install spacer`

#### Local clone
* Clone this repo
* Create a virtualenv
* pip install -r requirements.txt

### Code coverage
If you are using the docker build or local install, 
you can check code coverage like so:
 
1) Generate data
```
    coverage run --source=spacer --omit=spacer/tests/* -m unittest
``` 
2) Render simple report
```    
    coverage report -m
```    
3) Render to html
```
    coverage html
```
which renders html files to `.htmlcov`.