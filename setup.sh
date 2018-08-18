#!/bin/bash

# Exit upon if one command fails
set -e

# Exit if one variable is uninitialised
set -u

# First clone tensorflow models:
git clone https://github.com/tensorflow/models.git

# build all protos
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cd -

# Now install cocoapi
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ../../models/research/
cd -

# Now clone deeper-traffic lights:
git clone https://github.com/kenshiro-o/deeper-traffic-lights.git

cd deeper-traffic-lights
mkdir models
cd models
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz 
tar -xzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
rm ssd_mobilenet_v1_coco_2018_01_28.tar.gz
cd ..

# Create the bosch dataset
mkdir -p data/bosch
cp models/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config data/bosch/pipeline.config 
cp data_samples/label_map.pbtxt data/bosch/label_map.pbtxt

# Back to parent directory
cd ..
