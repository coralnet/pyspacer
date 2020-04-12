# This Dockerfile is modified from mapler/caffe-py3:cpu.
# Normally, one would inherit from that definition.
# But even mapler/caffe-py3:cpu was compiled with CUDA,
# so it doesn't build on systems without GPUs.
# So instead this build the image from ubuntu.
# Note that mapler/caffe-py3 built from nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
# Here I instaed start from the Ubuntu 18:04 base image.


# FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
# LABEL maintainer maplerme@gmail.com

FROM ubuntu:18.04
LABEL maintainer oscar.beijbom@gmail.com

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install tzdata

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
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
        python-numpy \
        python3-pip \
        python3-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
ENV CLONE_TAG=1.0

# Reduce caffe logging to not spam the console.
ENV GLOG_minloglevel=2

WORKDIR $CAFFE_ROOT

# Note that caffe setup will PIP install some python packages.
# These are required for the caffe compiler.
# Once caffe is compiled we upgrade packages as listed below.
RUN pip3 install --upgrade pip
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    cd python && for req in $(cat requirements.txt) pydot 'python-dateutil>2'; \
    do pip3 install $req; done && cd .. && \
    mkdir build && cd build && \
    cmake -DCPU_ONLY=1 -Dpython_version=3 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# Starting from here are spacer-specific commands.
# These could be run from requirements.txt after the COPY below
# But by doing it explicitly the docker build can cache them for faster builds.
RUN pip3 install boto==2.49.0
RUN pip3 install wget==3.2
RUN pip3 install coverage==5.0.3
RUN pip3 install tqdm==4.43.0
RUN pip3 install fire==0.2.1
RUN pip3 install Pillow==6.2.0
RUN pip3 install numpy==1.18.1
RUN pip3 install scikit-learn==0.22.1
RUN pip3 install scikit-image==0.15.0
# RUN pip3 install torch==1.4.0
# RUN pip3 install torchvision==0.5.0

ENV SPACER_LOCAL_MODEL_PATH=/workspace/models
ENV PYTHONPATH="/workspace/spacer:${PYTHONPATH}"
WORKDIR /workspace
RUN mkdir models
COPY . spacer
WORKDIR spacer
CMD coverage run --source=spacer --omit=spacer/tests/* -m unittest; coverage report -m