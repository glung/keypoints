FROM gcr.io/tensorflow/tensorflow:0.11.0-devel

RUN add-apt-repository universe
RUN apt-get -qq -y install curl
RUN apt-get -qq update
RUN apt-get -qq -y install python-pandas
RUN apt-get -qq -y install python-matplotlib
RUN pip install --upgrade pip
RUN pip install -U scikit-learn
RUN pip install scikit-image
RUN pip install tensorflow
RUN pip install git+https://github.com/tflearn/tflearn.git
RUN pip install pandas --upgrade
