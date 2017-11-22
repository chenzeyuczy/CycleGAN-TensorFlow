#! /bin/bash

MODEL=pretrained/occlude2white.pb
INPUT_IMG=data/input/0003C4T0003F067.jpg
OUTPUT_IMG=data/output/0003C4T0003F067.jpg
IMAGE_SIZE=128

python inference.py --model ${MODEL} --input ${INPUT_IMG} --output ${OUTPUT_IMG} \
	--image_size ${IMAGE_SIZE}
