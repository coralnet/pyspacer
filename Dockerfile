FROM bvlc/caffe:cpu
MAINTAINER oscar oscar.beijbom@gmail.com

WORKDIR /workspace

RUN apt-get -y update
RUN apt-get -y upgrade
RUN pip install --upgrade pip
RUN PYTHONPATH=$PYTHONPATH:/workspace/spacer

COPY . spacer
RUN pip install -r spacer/requirements.txt
CMD cd spacer
CMD python -m unittest discover

