# TensorRT-Examples
An example of using Nvidia's TensorRT library for popular Deep Learning architectures (in progress)

## Models
The [CNN model](./alexnet.py) implemented is adapted from [this tensorflow tutorial](https://www.tensorflow.org/tutorials/deep_cnn). The source code from which it was adapted can be found [here](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10). 

The model contains two convolutional layers coupled with max pooling and normalization. Following the convolutions are two fully connected layers, concluding with a softmax layer for classification. 

## Data
Data used was the Cifar10 dataset. Details on how to download it and read it can be found in the previously mentioned tutorial. 

