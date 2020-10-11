# This Dockerfile is modified from mapler/caffe-py3:cpu.
# https://hub.docker.com/r/mapler/caffe-py3/
# Normally, one would inherit from that definition,
# but it seems mapler/caffe-py3:cpu was compiled with CUDA,
# so it didn't build on systems without GPUs.
# So instead I had to copy-paste his dockerfile.
# Ultimately I switched to building form the Ubuntu 18:04 base image,
# to get the latest ubuntu and also to have a cleaner trace.

FROM ubuntu:18.04
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


# Install caffe.
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
RUN pip3 install wget==3.2
RUN pip3 install coverage==5.0.3
RUN pip3 install tqdm==4.43.0
RUN pip3 install fire==0.2.1
RUN pip3 install Pillow==6.2.0
RUN pip3 install numpy==1.18.1
RUN pip3 install scikit-learn==0.22.1
RUN pip3 install scikit-image==0.15.0
RUN pip3 install torch==1.4.0
RUN pip3 install torchvision==0.5.0
RUN pip3 install boto3==1.15.8
RUN pip3 install botocore====1.18.8

ENV SPACER_LOCAL_MODEL_PATH=/workspace/models
ENV PYTHONPATH="/workspace/spacer:${PYTHONPATH}"
WORKDIR /workspace
RUN mkdir models
RUN mkdir spacer
COPY ./spacer spacer/spacer
WORKDIR spacer

CMD coverage run --source=spacer --omit=spacer/tests/* -m unittest; coverage report -m
