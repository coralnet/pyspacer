# Use the official Python 3.10 image as a parent image
FROM python:3.10-slim

# Set environment variables to reduce Python package issues and ensure output is sent straight to the terminal without buffering it first
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory to /app
WORKDIR /app

# Copy main.py to /app
COPY main.py .

# Copy secrets.json to /app
COPY secrets.json .

# Install any needed packages specified in requirements.txt
# Assuming pyspacer is available on PyPI and has a requirements.txt file
# If pyspacer is not on PyPI, you might need to copy it into the container and install it manually
RUN pip install --no-cache-dir pyspacer

# Run main.py when the container launches
CMD ["python", "main.py"]


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
RUN for req in $(cat requirements.txt) pydot 'numpy==1.24.1' 'protobuf<4'; \
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
RUN pip3 install coverage==7.0.5
RUN pip3 install tqdm==4.65.0
RUN pip3 install fire==0.5.0
RUN pip3 install Pillow==10.1.0
RUN pip3 install scikit-learn==1.1.3
RUN pip3 install torch==1.13.1
RUN pip3 install torchvision==0.14.1
RUN pip3 install boto3==1.26.122

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
