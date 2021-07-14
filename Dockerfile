FROM tensorflow/tensorflow:2.3.0

WORKDIR=/tmp/

COPY Train_MNIST_TPU.py /tmp/
