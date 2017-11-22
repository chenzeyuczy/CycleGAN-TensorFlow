#! /bin/bash

CKPT_FILE=checkpoints/20171121-2225/model.ckpt-180000
X2Y_model=occlude2white.pb
Y2X_model=white2occlude.pb
image_size=128

python export_graph.py --ckpt ${CKPT_FILE} --XtoY_model ${X2Y_model} --YtoX_model \
    ${Y2X_model} --image_size ${image_size}

