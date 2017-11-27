#! /usr/bin/env python

import os
from shutil import copyfile

def merge_two_dir(dir1, dir2, output_dir, prefix1='', prefix2=''):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	filenames = os.listdir(dir1)
	filenames2 = os.listdir(dir2)
	for filename in filenames:
		if filename not in filenames2:
			print('{} not found in {}.'.format(filename, dir2))
		path_src = os.path.join(dir1, filename)
		path_dst = os.path.join(output_dir, prefix1 + filename)
		copyfile(path_src, path_dst)
		path_src = os.path.join(dir2, filename)
		path_dst = os.path.join(output_dir, prefix2 + filename)
		copyfile(path_src, path_dst)

if __name__ == '__main__':
	occlude_dir = '/home/zeyu/data/MARS/noise_multipatch/occlude'
	raw_dir = '/home/zeyu/data/MARS/noise_multipatch/whole'
	patch_dir = '/home/zeyu/data/MARS/noise_multipatch/single_raw_patch'
	merge_two_dir(occlude_dir, raw_dir, patch_dir, 'occlude_', 'raw_')

	label_dir = '/home/zeyu/data/MARS/noise_multipatch/white'
	raw_dir = '/home/zeyu/data/MARS/noise_multipatch/whole'
	patch_dir = '/home/zeyu/data/MARS/noise_multipatch/single_raw_label'
	merge_two_dir(occlude_dir, raw_dir, patch_dir, 'occlude_', 'raw_')

