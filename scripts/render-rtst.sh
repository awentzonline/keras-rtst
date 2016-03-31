#!/bin/bash
OUTPUT_PREFIX=$1
WEIGHTS_PREFIX=$2
CONVERT_GLOB=$3
VGG_WEIGHTS=$4
MAX_WIDTH=${MAX_WIDTH-256}

mkdir -p `dirname $OUTPUT_PREFIX`
echo "Converting files named: $CONVERT_GLOB"
rtst.py \
  $OUTPUT_PREFIX \
  --convert-glob=$CONVERT_GLOB \
  --vgg-weights=$VGG_WEIGHTS \
  --max-width=$MAX_WIDTH \
  --weights-prefix=$WEIGHTS_PREFIX
