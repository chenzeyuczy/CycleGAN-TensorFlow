#1 /bin/bash

image_size=256
DATA_ROOT=/home/zeyu/data/MARS/noise_pure
X=${DATA_ROOT}/occlude.tfrecords
Y=${DATA_ROOT}/label.tfrecords

python train.py --X ${X} --Y ${Y} --image_size ${image_size}
