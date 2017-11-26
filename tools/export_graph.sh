#! /bin/bash

ITER=1  # Time of iterations.
CKPT_FILE=checkpoints/20171125-2201/model.ckpt-${ITER}0000
X2Y_model=occlude2white_voc_full_${ITER}w.pb
Y2X_model=white2occlude_voc_full_${ITER}w.pb
image_size=256

python export_graph.py --ckpt ${CKPT_FILE} --XtoY_model ${X2Y_model} --YtoX_model \
    ${Y2X_model} --image_size ${image_size}

