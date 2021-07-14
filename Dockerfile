FROM tensorflow/tensorflow:2.3.0

RUN pip install tf-models-official==2.3.0 tensorboard-plugin-profile==2.3.0 cloud-tpu-client

WORKDIR /tmp/

COPY Train_MNIST_TPU.py /tmp/
