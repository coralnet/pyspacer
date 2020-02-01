FROM bvlc/caffe:cpu
MAINTAINER oscar oscar.beijbom@gmail.com

RUN apt-get -y update
RUN apt-get -y upgrade
RUN pip install --upgrade pip
COPY . spacer
RUN pip install -r spacer/requirements.txt
CMD cd spacer
CMD python -m unittest discover

