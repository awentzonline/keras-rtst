#!/bin/bash
# Download a tiny tiny tiny subset of imagenet
IMAGENET_URLS=/tmp/ilsvrc12_urls.txt.gz
NUM_IMAGES=${NUM_IMAGES-100}
echo "Nano ImageNet downloader"
# From this tutorial: http://www.deepdetect.com/tutorials/train-imagenet/
if [ ! -f $IMAGENET_URLS ]
then
  echo "Downloading the list of image URLs"
  curl http://www.deepdetect.com/dd/datasets/imagenet/ilsvrc12_urls.txt.gz -o $IMAGENET_URLS
fi
echo "Downloading some images..."
gunzip -c $IMAGENET_URLS | shuf -n $NUM_IMAGES | cut -f2 | xargs -n 1 curl -O --max-time 10
