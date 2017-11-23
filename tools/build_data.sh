#! /bin/bash

data_root='/home/zeyu/data/MARS/noise_multipatch'
X_input_dir=${data_root}'/occlude'
Y_input_dir=${data_root}'/white'
X_output_file=${data_root}'/occlude.tfrecords'
Y_output_file=${data_root}'/white.tfrecords'

python build_data.py --X_input_dir ${X_input_dir} --Y_input_dir ${Y_input_dir} \
	--X_output_file ${X_output_file} --Y_output_file ${Y_output_file}
