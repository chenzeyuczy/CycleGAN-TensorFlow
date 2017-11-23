#1 /bin/bash

image_size=256
X=/home/zeyu/data/MARS/noise_multipatch/occlude.tfrecords
Y=/home/zeyu/data/MARS/noise_multipatch/white.tfrecords

python train.py --X ${X} --Y ${Y} --image_size ${image_size}
