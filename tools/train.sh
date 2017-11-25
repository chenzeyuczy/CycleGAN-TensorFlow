#1 /bin/bash

image_size=256
DATA_ROOT=/home/zeyu/data/MARS/mix_voc_fullsize
X=${DATA_ROOT}/mix.tfrecords
Y=${DATA_ROOT}/label.tfrecords

GPU_DEVICE=1
#GPU_FRACTION=1.0

OPTION=""

if [[ -n "GPU_FRACTION" ]]; then
	OPTION="$OPTION --GPU_FRACTION ${GPU_FRACTION}"
fi

python train.py --X ${X} --Y ${Y} --image_size ${image_size} ${OPTION}
