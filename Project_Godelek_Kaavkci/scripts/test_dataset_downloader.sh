#!/bin/sh
directory=./data/test

mkdir -p $directory

#https://github.com/fperazzi/proSR/blob/main/data/get_data.sh
wget -P $directory 'http://cv.snu.ac.kr/research/EDSR/benchmark.tar'

for file in $directory/*.tar
do
    tar -xvf $file -C $directory && rm $file 
done
