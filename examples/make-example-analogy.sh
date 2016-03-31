# Train a texturizer using a mix of training images and noise.
# usage: ./make-example-texturizer.sh EXAMPLE_STYLE PATH_TO_TRAINING_IMAGES VGG_WEIGHTS
STYLE_NAME=$1
TRAINING_DATA=$2
EVAL_DATA=$3
VGG_WEIGHTS=${4-vgg16_weights.h5}
OUTPUT_PREFIX=out/$STYLE_NAME-analogy/render
WEIGHTS_PREFIX=$STYLE_NAME
STYLE_PATH=images/$STYLE_NAME.jpg
STYLE_MAP_PATH=images/$STYLE_NAME-a.jpg
rtst.py \
  $OUTPUT_PREFIX \
  --style-img=$STYLE_PATH \
  --style-map-img=$STYLE_MAP_PATH \
  --train \
  --max-width=${MAX_WIDTH-128} \
  --content-w=${CONTENT_W-0.0} \
  --style-w=${STYLE_W-0.0} \
  --analogy-w=${ANALOGY_W-1.0} \
  --mrf-w=${MRF_W-1.0} \
  --batch-size=1 \
  --weights-prefix=$WEIGHTS_PREFIX \
  --vgg-weights=$VGG_WEIGHTS \
  --train-data=$TRAINING_DATA \
  --eval-data=$EVAL_DATA \
  --ignore --auto-save-weights
