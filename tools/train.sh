#! /bin/bash

image_size=256
DATA_ROOT=/home/zeyu/data/MARS/mix_imagenet_refined_gradient
X=${DATA_ROOT}/occlude.tfrecords
Y=${DATA_ROOT}/label.tfrecords

#MODEL=checkpoints/20171207-1339
# MODEL_ITER=30000
# MODEL_ITER=30000

#GPU_FRACTION=0.8

OPTION=""

if [[ -n "${GPU_FRACTION}" ]]; then
	OPTION="${OPTION} --gpu_fraction ${GPU_FRACTION}"
fi
if [[ -n "${MODEL}" ]]; then
	OPTION="${OPTION} --load_model ${MODEL}"
fi
if [[ -n "${MODEL_ITER}" ]]; then
	OPTION="${OPTION} --model_iter ${MODEL_ITER}"
fi

python train.py --X ${X} --Y ${Y} --image_size ${image_size} ${OPTION}

