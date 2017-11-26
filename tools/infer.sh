#! /bin/bash

ITER=1  # Time of iterations.
MODEL=pretrained/occlude2white_voc_full_${ITER}w.pb
INPUT_DIR=data/input/occluded_body_images
OUTPUT_DIR=data/output/voc_full_${ITER}w
IMAGE_SIZE=256

python batch_inference.py --model ${MODEL} --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} \
	--image_size ${IMAGE_SIZE}
