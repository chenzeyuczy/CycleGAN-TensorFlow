#! /usr/bin/env python

import os
from PIL import Image
import numpy as np

data_root = '/home/zeyu/data/MARS/mix_cuhk01_delicate'

def image_padding(img):
# Resize image to square.
	h, w, _ = img.shape  # Size of image: (128, 64, 3)
	padding = (h - w) / 2  # height > width
	side = h
	img_target = np.zeros(shape=(side, side, 3), dtype=img.dtype)
	img_target[:, padding: padding + w, :] = img
	return img_target

dir_list = ['occlude', 'label']
for dir_name in dir_list:
	dir_path_src = os.path.join(data_root, dir_name)
	dir_path_dst = os.path.join(data_root, dir_name + '_padding')

	if not os.path.exists(dir_path_dst):
		os.makedirs(dir_path_dst)
	file_list = os.listdir(dir_path_src)

	num_file = len(file_list)
	count_file = 0
	for file_name in file_list:
		file_path_src = os.path.join(dir_path_src, file_name)
		file_path_dst = os.path.join(dir_path_dst, file_name)

		img = np.array(Image.open(file_path_src))
		img_padding = image_padding(img)
		Image.fromarray(img_padding).save(file_path_dst)

		count_file += 1
		if count_file % 1000 == 0:
			print('%d / %d images in directory %s done.' % (count_file, num_file, dir_name))

	print('Images in directory %s have been padded.' % dir_name)

