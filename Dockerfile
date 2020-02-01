FROM bvlc/caffe:cpu
MAINTAINER oscar oscar.beijbom@gmail.com

RUN apt-get -y update
RUN apt-get -y upgrade
RUN pip install --upgrade pip

WORKDIR /root/.aws
COPY secrets credentials

WORKDIR /workspace
COPY . spacer
RUN PYTHONPATH=$PYTHONPATH:/workspace/spacer

WORKDIR spacer
RUN pip install -r requirements.txt
CMD python -m unittest discover

