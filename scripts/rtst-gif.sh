#!/bin/bash
# Styles a gif by splitting out the frames and processing them individually.
# Uses `convert` tool from ImageMagick http://www.imagemagick.org/script/binary-releases.php
GIF=$1
WEIGHTS_PREFIX=$2
OUTPUT_PREFIX=$3
VGG_WEIGHTS=${4-vgg16_weights.h5}
MAX_WIDTH=${MAX_WIDTH-512}
FRAMES_PATH=$OUTPUT_PREFIX/frames
PROCESSED_PATH=$OUTPUT_PREFIX/processed
DELAY=5
CONVERT_GLOB="$FRAMES_PATH/*.png"

echo "Texturizing a gif."
echo "Splitting $GIF..."
mkdir -p $FRAMES_PATH
mkdir -p $PROCESSED_PATH
convert -alpha Remove -coalesce $GIF $FRAMES_PATH/%04d.png

echo "Texturizing frames...$CONVERT_GLOB"
rtst.py \
  $PROCESSED_PATH \
  --weights-prefix=$WEIGHTS_PREFIX \
  --convert-glob=$CONVERT_GLOB \
  --vgg-weights=$VGG_WEIGHTS \
  --max-width=$MAX_WIDTH

echo "Combining new frames..."
convert -delay $DELAY -loop 0 $PROCESSED_PATH/*.png $PREFIX/result.gif
