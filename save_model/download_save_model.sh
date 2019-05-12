#!/bin/bash

# Download integrated zip file (refer for windows, coco api for windows)
wget -c https://www.dropbox.com/s/5yd07fezdamkbqc/save_model.zip
wget -c https://www.dropbox.com/s/grjgbbo89hnm8l7/resnet50-pretrained.pth
wget -c https://www.dropbox.com/s/qqmqwxe6ik83jrx/resnet34-pretrained.pth

unzip -q save_model.zip
rm -rf save_model.zip
