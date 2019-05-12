#!/bin/bash

# Download integrated zip file (refer for windows, coco api for windows)
wget -c https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

tar xvzf cifar-100-python.tar.gz
rm -rf cifar-100-python.tar.gz