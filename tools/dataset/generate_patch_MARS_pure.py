#! /usr/bin/env python

import os
import numpy as np
from PIL import Image

# Setup configuration
root_src = '/home/zeyu/data/MARS/bbox_train'
root_dst = '/home/zeyu/data/MARS/noise_pure'
root_occlude = os.path.join(root_dst, 'occlude')
root_label = os.path.join(root_dst, 'label')
label_color = (255, 255, 255)

for path_dir in [root_occlude, root_label]:
	if not os.path.exists(path_dir):
		os.makedirs(path_dir)

root_dirs = os.listdir(root_src)
num_dir = len(root_dirs)
count_dir = 0

for root_dir in root_dirs:
	path_dir = os.path.join(root_src, root_dir)
	filenames = os.listdir(path_dir)

	# Select part of images from folder at random.
	num_file = len(filenames)
	max_num_file = 50
	np.random.shuffle(filenames)
	for filename in filenames[:min(num_file, max_num_file)]:
		path_src = os.path.join(path_dir, filename)
		path_occlude = os.path.join(root_occlude, filename)
		path_label = os.path.join(root_label, filename)

		img_src = np.array(Image.open(path_src))
		[H, W, _] = img_src.shape
		img_occlude = img_src.copy()
		img_label = img_src.copy()

		occlude_type = np.random.randint(2)
		occlude_num = np.random.randint(2) + 1
		occlude_split = np.random.randint(2)
		occlude_pos = np.random.randint(2)

		# Get bounding box of occluding patch.
		# Horizonal occlusion.
		if occlude_type == 0:
			patch_w = np.random.randint(int(W * 0.6), W)
			patch_h = np.random.randint(int(H * 0.3), int(H * 0.5))
			patch_x = np.random.randint(W - patch_w)
			patch_y = np.random.randint(int(H * 0.1)) + int(H * 0.9 - patch_h) * occlude_pos
		# Vertical occlusion.
		else:
			patch_w = np.random.randint(int(W * 0.2), int(W * 0.4)) 
			patch_h = np.random.randint(int(H * 0.6), H)
			patch_x = np.random.randint(int(W * 0.1)) + int(W * 0.9 - patch_w) * occlude_pos
			patch_y = np.random.randint(H - patch_h)

		start_x, start_y = patch_x, patch_y
		for occlude_idx in range(occlude_num):
			occlude_color = np.random.randint(255, size=3)
			# Horizonal split.
			if occlude_split == 0:
				if occlude_idx == occlude_num - 1:
					increment = patch_x + patch_w - start_x
				else:
					increment = np.random.randint(patch_x + patch_w - start_x)
				img_occlude[patch_y: patch_y + patch_h, start_x: start_x + increment, :] = occlude_color
				start_x += increment
			# Vertical split
			else:
				if occlude_idx == occlude_num - 1:
					increment = patch_y + patch_h - start_y
				else:
					increment = np.random.randint(patch_y + patch_h - start_y)
				img_occlude[start_y: start_y + increment, patch_x: patch_x + patch_w, :] = occlude_color
				start_y += increment
		img_label[patch_y: patch_y + patch_h, patch_x: patch_x + patch_w, :] = label_color
		
		# Save images.
		Image.fromarray(img_occlude).save(path_occlude)
		Image.fromarray(img_label).save(path_label)

	# Display progress
	count_dir += 1
	if count_dir % 100 == 0:
		print('{}/{} folders finished.'.format(count_dir, num_dir))

print('All images generated.')

