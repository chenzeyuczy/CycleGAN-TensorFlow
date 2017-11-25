#! /usr/bin/env python

import os
import numpy as np
from PIL import Image
from scipy import misc
from skimage import filters as imfilter

def find_min_box(img):
# Find minimal bounding box of object within a gray-mode image.
	[H, W] = img.shape
	left, right, top, bottom = W, -1, H, -1
	for i in xrange(H):
		for j in xrange(W):
			if img[i, j] != 255 and img[i, j] != 0:
				left = j if left > j else left
				right = j if right < j else right
				top = i if top > i else top
				bottom = i if bottom < i else bottom
	x, y, w, h = left, top, right - left + 1, bottom - top + 1
	return (x, y, w, h)

def mix_image(img_src, img_occlude_raw, img_occlude_anno, patch_box):
# Function to mix original image with occluding image.
	label_color = 255
	patch_x, patch_y, patch_w, patch_h = patch_box
	# Resize patch to target size.
	min_box = find_min_box(img_occlude_anno)
	x, y, w, h = min_box
	img_occlude_raw = img_occlude_raw[y:y + h, x:x + w, :]
	img_occlude_anno = img_occlude_anno[y:y + h, x:x + w]

	# Apply gaussian blur on raw image.
	gaussian_sigma = 3
	img_occlude_raw = imfilter.gaussian(img_occlude_raw, sigma=gaussian_sigma, multichannel=True)
	img_occlude_anno = imfilter.gaussian(img_occlude_anno, sigma=gaussian_sigma, multichannel=True)

	# Resize occlusion blocks.
	img_occlude_raw = misc.imresize(img_occlude_raw, (patch_h, patch_w))
	img_occlude_anno = misc.imresize(img_occlude_anno, (patch_h, patch_w))
	img_mix = img_src.copy()
	img_label = img_src.copy()

	[H, W, _] = img_src.shape
	# Coordinate of anchor to be added in ooriginal image.
	offset_raw_x = max(0, patch_x)
	offset_raw_y = max(0, patch_y)
	# Coordinate of anchor start to be add in patch.
	offset_patch_x = max(0, -patch_x)
	offset_patch_y = max(0, -patch_y)
	# Actual width and height of patch to be add.
	actual_w = min(patch_x + patch_w, W) - offset_raw_x
	actual_h = min(patch_y + patch_h, H) - offset_raw_y

	for i in xrange(actual_h):
		for j in xrange(actual_w):
			# Case of annotation region.
			if img_occlude_anno[offset_patch_y + i, offset_patch_x + j] != 255 and \
				img_occlude_anno[offset_patch_y + i, offset_patch_x + j] != 0:
				img_mix[offset_raw_y + i, offset_raw_x + j, :] = img_occlude_raw[offset_patch_y + i, offset_patch_x + j, :]
				img_label[offset_raw_y + i, offset_raw_x + j, :] = label_color

#	for i in range(patch_h):
#		for j in range(patch_w):
#			# Case of annotation region.
#			if img_occlude_anno[i, j] != 255 and \
#				img_occlude_anno[i, j] != 0:
#				img_mix[patch_y + i, patch_x + j, :] = img_occlude_raw[i, j, :]
#				img_label[patch_y + i, patch_x + j, :] = label_color
	return img_mix, img_label

def main():
	# Setup path and some other parameters.
	src_root = '/home/zeyu/data/MARS/bbox_train'
	occlude_root = '/home/zeyu/data/PascalVOC2007/VOCdevkit/VOC2007'
	dst_root = '/home/zeyu/data/MARS/mix_voc_fullsize'

	MAX_NUM = 10000
	patch_w_ratio_min = 0.8
	patch_w_ratio_max = 1.2
	patch_h_ratio_min = 0.8
	patch_h_ratio_max = 1.2
	margin_ratio_max = 0.5
	
	src_folder = os.listdir(src_root)
	occlude_raw_folder = os.path.join(occlude_root, 'JPEGImages')
	occlude_gt_folder = os.path.join(occlude_root, 'JPEG_Segmentation')
	occlude_raw_files = os.listdir(occlude_raw_folder)
	occlude_files = os.listdir(occlude_gt_folder)

	folder_mix = os.path.join(dst_root, 'mix')
	folder_label = os.path.join(dst_root, 'label')
	if not os.path.exists(folder_mix):
		os.makedirs(folder_mix)
	if not os.path.exists(folder_label):
		os.makedirs(folder_label)

	for img_counter in xrange(MAX_NUM):
		# Select occlude file.
		occlude_idx = np.random.randint(len(occlude_files))
		occlude_file = occlude_files[occlude_idx]
		file_path_anno = os.path.join(occlude_gt_folder, occlude_file)
		file_path_raw = os.path.join(occlude_raw_folder, occlude_file[:-4] + '.jpg')
		img_anno = np.array(Image.open(file_path_anno))
		img_raw = np.array(Image.open(file_path_raw))
		
		# Select image to be occlude.
		src_folder_idx = np.random.randint(len(src_folder))
		src_sub_folder = os.path.join(src_root, src_folder[src_folder_idx])
		src_img_list = os.listdir(src_sub_folder)
		src_img_idx = np.random.randint(len(src_img_list))
		src_img_path = os.path.join(src_sub_folder, src_img_list[src_img_idx])
		img_src = np.array(Image.open(src_img_path))
	
		# Select patch location at random.
		[H, W, _] = img_src.shape
		patch_w = np.random.randint(round(W * patch_w_ratio_min), round(W * patch_w_ratio_max))
		patch_h = np.random.randint(round(H * patch_h_ratio_min), round(H * patch_h_ratio_max))
		patch_x = np.random.randint(round(-W * margin_ratio_max), round(W * (1 +	margin_ratio_max)) - patch_w)
		patch_y = np.random.randint(round(-H * margin_ratio_max), round(H * (1 +	margin_ratio_max)) - patch_h)
		patch_box = (patch_x, patch_y, patch_w, patch_h)
	
		# Generate mixed image pair.
		img_mix, img_label = mix_image(img_src, img_raw, img_anno, patch_box)
	
		# Save result.
		name_src = os.path.splitext(os.path.basename(src_img_path))[0]
		name_occlude = os.path.splitext(os.path.basename(occlude_file))[0]
		file_mix = '_'.join(['mix', name_src, name_occlude, '-'.join(map(str, patch_box))]) + '.jpg'
		file_label = '_'.join(['label', name_src, name_occlude,	'-'.join(map(str, 	patch_box))]) + '.jpg'
		path_mix = os.path.join(folder_mix, file_mix)
		path_label = os.path.join(folder_label, file_label)
		Image.fromarray(img_mix).save(path_mix)
		Image.fromarray(img_label).save(path_label)

		# Display progress.
		if img_counter % 100 == 99:
			print('{} / {} images have been generated.'.format(img_counter + 1, MAX_NUM))

if __name__ == '__main__':
	main()

