#! /bin/bash

DEFAULT_ITER=1 # Time of iterations.

# Get arguments from command line.
if [[ "$#" > 1 ]]
then
	IDX_BEGIN=$1
	IDX_END=$2
else
	IDX_BEGIN=${DEFAULT_ITER}
	IDX_END=${DEFAULT_ITER}
fi

# Process in a loop.
for ITER in `seq ${IDX_BEGIN} ${IDX_END}`
do
	MODEL_NAME=voc_gray
	MODEL=pretrained/occlude2gray_${MODEL_NAME}_${ITER}w.pb
	INPUT_DIR=data/input/occluded_body_images
	OUTPUT_DIR=data/output/${MODEL_NAME}_${ITER}w
	IMAGE_SIZE=256

	python batch_inference.py --model ${MODEL} --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} \
		--image_size ${IMAGE_SIZE}
done
