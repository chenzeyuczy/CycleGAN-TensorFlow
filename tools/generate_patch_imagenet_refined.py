#! /usr/bin/env python

import os
import numpy as np
from PIL import Image
from scipy.misc import imresize
from skimage import filters as imfilter

input_root = '/home/zeyu/data/MARS/bbox_train'
patch_root = '/home/zeyu/data/imagenet_occlusion'
output_dir = '/home/zeyu/data/MARS/mix_imagenet_refined_delicate'
label_color = 255
label_color = 127

# Occlusion type: 0-uppper half; 1-lower half; 2-left half; 3-right half
occlusion_types = {'stem': [2, 3], 'office': [1], 'umbrella': [0, 1], 'wall': [1, 2, 3], 'plant':
[0, 1], 'leaf': [0, 1], 'chair': [1], 'door': [2, 3], 'vehicle': [1], 'person': [2, 3]}

input_dirs = os.listdir(input_root)
patch_dirs = os.listdir(patch_root)
num_input_dir = len(input_dirs)
num_patch_dir = len(patch_dirs)

occlude_dir = os.path.join(output_dir, 'occlude')
label_dir = os.path.join(output_dir, 'label')
raw_dir = os.path.join(output_dir, 'raw')

# Create output directories.
dir_paths = [occlude_dir, label_dir, raw_dir]
for dir_path in dir_paths:
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)

# Preload file lists.
patch_files = []
for i in xrange(num_patch_dir):
	patch_dir_path = os.path.join(patch_root, patch_dirs[i])
	patch_files.append(os.listdir(patch_dir_path))

max_img_per_dir = 40
count_dir = 0
for input_dir in input_dirs:
	input_dir_path = os.path.join(input_root, input_dir)
	input_files = os.listdir(input_dir_path)
	np.random.shuffle(input_files)
	for input_file in input_files[:max_img_per_dir]:
		input_path = os.path.join(input_dir_path, input_file)

		while True:
			patch_dir_idx = np.random.randint(num_patch_dir)
			patch_type = patch_dirs[patch_dir_idx]
			occlusion_type = np.random.choice(occlusion_types[patch_type])

			patch_dir = patch_dirs[patch_dir_idx]
			patch_file_idx = np.random.randint(len(patch_files[patch_dir_idx]))
			patch_file = patch_files[patch_dir_idx][patch_file_idx]
			patch_path = os.path.join(patch_root, patch_dir, patch_file)

			img_patch = np.array(Image.open(patch_path))
			if img_patch.shape[-1] == 3:
				break

		img_input = np.array(Image.open(input_path))
		H, W, _ = img_input.shape

		# Apply gaussian blur on patch image.
		gaussian_sigma = 3
		img_patch = imfilter.gaussian(img_patch, sigma=gaussian_sigma, multichannel=True)

		img_occlude = img_input.copy()
		img_label = img_input.copy()
		img_alpha = np.ones(img_input.shape, dtype=np.float64)
#		img_label = np.zeros(img_input.shape, dtype=np.uint8)

		occlude_pos = occlusion_type % 2

		# Get bounding box of occluding patch.
		# Horizonal occlusion.
		if occlusion_type < 2:
			patch_w = np.random.randint(int(W * 0.7), W)
			patch_h = np.random.randint(int(H * 0.3), int(H * 0.5))
			patch_x = np.random.randint(W - patch_w)
			patch_y = np.random.randint(int(H * 0.5 - patch_h)) + int(H * 0.5) * occlude_pos
		# Vertical occlusion.
		else:
			patch_w = np.random.randint(int(W * 0.2), int(W * 0.4)) 
			patch_h = np.random.randint(int(H * 0.7), H)
			patch_x = np.random.randint(int(W * 0.5 - patch_w)) + int(W * 0.5) * occlude_pos
			patch_y = np.random.randint(H - patch_h)

		img_occlude[patch_y: patch_y + patch_h, patch_x: patch_x + patch_w, :] = \
			imresize(img_patch, [patch_h, patch_w])
		img_alpha[patch_y: patch_y + patch_h, patch_x: patch_x + patch_w, :] = 0

		gaussian_sigma = 9
		img_alpha = imfilter.gaussian(img_alpha, sigma=gaussian_sigma, multichannel=True)
		img_label = img_input.astype(np.float64) * img_alpha + label_color * (1 - img_alpha)
		img_label = img_label.astype(np.uint8)

		occlude_path = os.path.join(occlude_dir, input_file)
		label_path = os.path.join(label_dir, input_file)
		raw_path = os.path.join(raw_dir, input_file)

		Image.fromarray(img_occlude).save(occlude_path)
		Image.fromarray(img_label).save(label_path)
		Image.fromarray(img_input).save(raw_path)

	# Display progress.
	count_dir += 1
	if count_dir % 100 == 0:
		print('{}/{} directories have been processed.'.format(count_dir, num_input_dir))

