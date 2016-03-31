#!/bin/bash
# Train a texturizer using a mix of training images and noise.
# usage: ./make-example-texturizer.sh EXAMPLE_STYLE PATH_TO_TRAINING_IMAGES VGG_WEIGHTS
STYLE_SRC=$1
STYLE_NAME=$(basename "$STYLE_SRC")
TRAINING_DATA=$2
EVAL_DATA=$3
VGG_WEIGHTS=${4-vgg16_weights.h5}
OUTPUT_PREFIX=$STYLE_NAME-rtst/render
WEIGHTS_PREFIX=$STYLE_NAME
STYLE_PATH=$STYLE_SRC

echo "Training RTST output at $OUTPUT_PREFIX"
rtst.py \
  $OUTPUT_PREFIX \
  --style-img=$STYLE_SRC \
  --train \
  --max-width=${MAX_WIDTH-256} \
  --content-w=${CONTENT_W-1} \
  --content-layers=${CONTENT_LAYERS-conv2_2} \
  --style-w=${STYLE_W-10.0} \
  --weights-prefix=$WEIGHTS_PREFIX \
  --vgg-weights=$VGG_WEIGHTS \
  --train-data=$TRAINING_DATA \
  --eval-data=$EVAL_DATA \
  --auto-save-weights
