#! /bin/bash

data_root='/home/zeyu/data/MARS/noise'
X_input_dir=${data_root}'/single_raw_patch'
Y_input_dir=${data_root}'/single_raw_label'
X_output_file=${data_root}'/occlude_non.tfrecords'
Y_output_file=${data_root}'/label_both.tfrecords'

python build_data.py --X_input_dir ${X_input_dir} --Y_input_dir ${Y_input_dir} \
	--X_output_file ${X_output_file} --Y_output_file ${Y_output_file}
