#1 /bin/bash

image_size=256
DATA_ROOT=/home/zeyu/data/MARS/mix_voc
X=${DATA_ROOT}/mix.tfrecords
Y=${DATA_ROOT}/label.tfrecords

python train.py --X ${X} --Y ${Y} --image_size ${image_size}
