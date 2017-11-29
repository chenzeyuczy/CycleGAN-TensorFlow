#! /bin/bash

ITER=9 # Time of iterations.
MODEL_NAME=voc_full
MODEL=pretrained/occlude2white_${MODEL_NAME}_${ITER}w.pb
INPUT_DIR=data/input/occluded_body_images
OUTPUT_DIR=data/output/${MODEL_NAME}_${ITER}w
IMAGE_SIZE=256

python batch_inference.py --model ${MODEL} --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} \
	--image_size ${IMAGE_SIZE}
