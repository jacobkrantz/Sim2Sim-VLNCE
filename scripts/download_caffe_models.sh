#!/bin/bash

# downloads the Caffe model to data/caffe_models

wget --directory-prefix=data/caffe_models https://raw.githubusercontent.com/peteanderson80/Matterport3DSimulator/master/models/ResNet-152-deploy.prototxt
wget --directory-prefix=data/caffe_models http://places2.csail.mit.edu/models_places365/resnet152_places365.caffemodel
