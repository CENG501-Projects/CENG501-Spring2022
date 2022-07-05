#! /bin/bash
mkdir -p mnist
# unpack tar
pushd mnist
    wget https://raw.githubusercontent.com/myleott/mnist_png/master/mnist_png.tar.gz
    tar -xzvf mnist_png.tar.gz
popd
# KMP_DUPLICATE_LIB_OK=TRUE python datasets.py cifar10 cifar10/cifar-10-batches-py -o cifar10/output
