#! /bin/bash
mkdir -p cifar10
# unpack tar
pushd cifar10
    wget  http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
popd
KMP_DUPLICATE_LIB_OK=TRUE python datasets.py cifar10 cifar10/cifar-10-batches-py -o cifar10/output
