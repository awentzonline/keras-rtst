# Train a texturizer using a mix of training images and noise.
# usage: ./make-example-texturizer.sh EXAMPLE_STYLE PATH_TO_TRAINING_IMAGES VGG_WEIGHTS
STYLE_NAME=$1
TRAINING_DATA=$2
EVAL_DATA=$3
VGG_WEIGHTS=${4-vgg16_weights.h5}
OUTPUT_PREFIX=out/$STYLE_NAME-xfer/render
WEIGHTS_PREFIX=$STYLE_NAME
STYLE_PATH=images/$STYLE_NAME.jpg

rtst.py \
  $OUTPUT_PREFIX \
  --style-img=$STYLE_PATH \
  --train \
  --max-width=${MAX_WIDTH-256} \
  --content-w=${CONTENT_W-0.01} \
  --content-layers=${CONTENT_LAYERS-conv2_2} \
  --style-w=${STYLE_W-10.0} \
  --weights-prefix=$WEIGHTS_PREFIX \
  --vgg-weights=$VGG_WEIGHTS \
  --train-data=$TRAINING_DATA \
  --eval-data=$EVAL_DATA \
  --model=girthy --depth=3 --num-res-filters=64 \
  --ignore --auto-save-weights
