#! /bin/bash

ITER=1 # Time of iterations.
CKPT_FILE=checkpoints/20171128-2137/model.ckpt-${ITER}0000
MODEL_PREFIX=voc_gray
X2Y_model=occlude2gray_${MODEL_PREFIX}_${ITER}w.pb
Y2X_model=gray2occlude_${MODEL_PREFIX}_${ITER}w.pb
image_size=256

python export_graph.py --ckpt ${CKPT_FILE} --XtoY_model ${X2Y_model} --YtoX_model \
    ${Y2X_model} --image_size ${image_size}

