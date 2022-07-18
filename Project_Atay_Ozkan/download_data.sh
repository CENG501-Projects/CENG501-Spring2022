#!/bin/bash
fileid="147DN5V8tB8GLxx3IgTwHxKXzuTfkE7qe"
filename="dataset.tar.xz"

CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=${fileid}" -O ${filename}
rm -f /tmp/cookies.txt
tar -xf dataset.tar.xz


