# This Dockerfile is modified from mapler/caffe-py3:cpu.
# https://hub.docker.com/r/mapler/caffe-py3/
#
# We copy-paste parts from that file instead of inheriting the whole thing,
# because:
# - mapler/caffe-py3:cpu was compiled with CUDA, so it doesn't build on systems
#   without GPUs.
# - We want a newer Ubuntu than 16.04, and several other installation details
#   change as a result.


# Ubuntu setup + Caffe installation
# Python 3.10 is the default Python for Ubuntu 22.04.
FROM ubuntu:22.04 as caffe
LABEL maintainer oscar.beijbom@gmail.com

# This section has to do with setting the time-zone and installing the OS.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install tzdata
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN dpkg-reconfigure --frontend noninteractive tzdata


# Install packages that we need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        vim \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-setuptools && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# Upgrade pip itself.
RUN pip3 install --upgrade pip
# Git-clone a fork of caffe which supports more modern software, particularly
# OpenCV 4, thus avoiding errors such as "'CV_LOAD_IMAGE_COLOR' was not
# declared in this scope" when building caffe.
ENV CLONE_TAG=ssd
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/Qengineering/caffe.git .

# pip-install python packages required for the caffe compiler.
#
# We can later change some package versions for spacer. However, numpy needs
# to stay the same after building caffe. Otherwise, it may get an error like
# "module compiled against API version 0x10 but this version of numpy is 0xe"
# when importing caffe.
#
# caffe's requirements.txt doesn't cap the protobuf version, but protobuf 4.x
# results in an error like "Couldn't build proto file into descriptor pool:
# duplicate file name" when importing caffe. So we avoid protobuf 4.x.
WORKDIR $CAFFE_ROOT/python
RUN for req in $(cat requirements.txt) pydot 'numpy==2.1.3' 'protobuf<4'; \
    do pip3 install $req; \
    done

# Reduce caffe logging to not spam the console.
ENV GLOG_minloglevel=2
# Build caffe.
WORKDIR $CAFFE_ROOT
RUN mkdir build
WORKDIR $CAFFE_ROOT/build
RUN cmake -DCPU_ONLY=1 -Dpython_version=3 ..
RUN make -j"$(nproc)"
# Set env vars and configure caffe.
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig


# Spacer installation
# We start a new Docker stage so that we can have a branch point after caffe
# and before spacer requirements, which can be useful for testing.
FROM caffe AS spacer

# These could be run from requirements.txt after the COPY below
# But by doing it explicitly, the docker build can cache each step's result
# for faster builds.
# Note that numpy is not here because it was specified before building caffe.
RUN pip3 install coverage==7.6.8
RUN pip3 install fire==0.7.0
RUN pip3 install Pillow==11.0.0
RUN pip3 install scikit-learn==1.5.2
RUN pip3 install torch==2.7.1
RUN pip3 install torchvision==0.22.1
RUN pip3 install boto3==1.34.162

ENV SPACER_EXTRACTORS_CACHE_DIR=/workspace/models
ENV PYTHONPATH="/workspace/spacer:${PYTHONPATH}"
WORKDIR /workspace
RUN mkdir spacer
COPY ./spacer spacer/spacer
WORKDIR spacer

# Run unit tests with code coverage reporting.
# - Exclude coverage of the unit tests themselves, to not reduce the score for
#   skipped tests.
# - Output coverage to a file outside of /workspacer/spacer, in case that dir
#   is read-only (preferable for some Docker setups).
# - Print coverage results.
CMD coverage run --data-file=/workspace/.coverage \
    --source=spacer --omit=spacer/tests/* -m unittest \
    && coverage report --data-file=/workspace/.coverage -m
