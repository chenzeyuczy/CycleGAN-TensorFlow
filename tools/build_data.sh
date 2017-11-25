#! /bin/bash

data_root='/home/zeyu/data/MARS/mix_voc_fullsize'
X_input_dir=${data_root}'/mix'
Y_input_dir=${data_root}'/label'
X_output_file=${data_root}'/mix.tfrecords'
Y_output_file=${data_root}'/label.tfrecords'

python build_data.py --X_input_dir ${X_input_dir} --Y_input_dir ${Y_input_dir} \
	--X_output_file ${X_output_file} --Y_output_file ${Y_output_file}
