FROM gcr.io/tensorflow/tensorflow:latest-devel

RUN add-apt-repository universe
RUN apt-get -qq -y install curl
RUN apt-get -qq update
RUN apt-get -qq -y install python-pandas
RUN apt-get -qq -y install python-matplotlib
RUN pip install --upgrade pip
RUN pip install -U scikit-learn
RUN pip install scikit-image

