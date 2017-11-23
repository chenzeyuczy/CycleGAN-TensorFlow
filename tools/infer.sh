#! /bin/bash

MODEL=pretrained/occlude2white.pb
INPUT_IMG=data/input/0017C4T0002F031.jpg
OUTPUT_IMG=data/output/0017C4T0002F031.jpg
IMAGE_SIZE=128

python inference.py --model ${MODEL} --input ${INPUT_IMG} --output ${OUTPUT_IMG} \
	--image_size ${IMAGE_SIZE}
