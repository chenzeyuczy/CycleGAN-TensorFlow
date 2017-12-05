#! /bin/bash

DEFAULT_ITER=1 # Time of iterations.

# Get arguments from command line.
if [[ "$#" > 1 ]]
then
	IDX_BEGIN=$1
	IDX_END=$2
else
	IDX_BEGIN=${DEFAULT_ITER}
	IDX_END=${DEFAULT_ITER}
fi

# Process in a loop.
for ITER in `seq ${IDX_BEGIN} ${IDX_END}`
do
	CKPT_FILE=checkpoints/20171201-1056/model.ckpt-${ITER}0000
	MODEL_PREFIX=voc_gray
	X2Y_model=occlude2gray_${MODEL_PREFIX}_${ITER}w.pb
	Y2X_model=gray2occlude_${MODEL_PREFIX}_${ITER}w.pb
	image_size=256

	python export_graph.py --ckpt ${CKPT_FILE} --XtoY_model ${X2Y_model} --YtoX_model \
		${Y2X_model} --image_size ${image_size}
done
