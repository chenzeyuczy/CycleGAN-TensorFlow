#! /bin/bash

data_root='/home/zeyu/data/MARS/mix_cuhk01_delicate'
X_input_dir=${data_root}'/occlude_padding'
Y_input_dir=${data_root}'/label_padding'
X_output_file=${data_root}'/occlude_padding.tfrecords'
Y_output_file=${data_root}'/label_padding.tfrecords'

python build_data.py --X_input_dir ${X_input_dir} --Y_input_dir ${Y_input_dir} \
	--X_output_file ${X_output_file} --Y_output_file ${Y_output_file}
