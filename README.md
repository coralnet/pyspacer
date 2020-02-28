# PySpacer

[![Build Status](https://travis-ci.org/beijbom/pyspacer.svg?branch=master)](https://travis-ci.org/beijbom/pyspacer)

This repository provide utilities to extract features from random point locations in images and then training classifiers over those features.
It is used heavily with `https://github.com/beijbom/coralnet`.

### Installation

Spacer has two installation modes.
The full package requires the deep learning framework `caffe` to be installed.
Since this can be a drag, caffe is only supported through docker.

#### Run including caffe using Docker
* Install docker on your system
* Build image: `docker build -t "test:Dockerfile" .`
* Run `docker run -v /path/to/local/folder/:/workspace/models -it test3:Dockerfile`

The `-v /path/to/local/folder/:/workspace/models` part will make sure the downloaded models are cached to your local disk (outside container), which makes rerunning stuff much faster.

This will run the default CMD command specified in the dockerfile (unit-test with coverage).
If you want to enter the docker container do: `docker run -it test3:Dockerfile bash`.


#### Run without caffe support using virtulenv.
* Install virtualenv
* `mkvirtualenv spacer --python /path/to/your/python3`
* `pip install -r requirements.txt`
* python -m unittest

### Code coverage

First generate report

    coverage run --source=spacer --omit=spacer/tests/* -m unittest
    
Render simple with
    
    coverage report -m
    
And to html with

    coverage html

which renders html files to `.htmlcov`.

