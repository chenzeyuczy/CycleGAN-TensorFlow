#! /usr/bin/env python

import numpy as np
from PIL import Image
import os

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

def segment_image(img):
	img = img_as_float(img)
	gradient = sobel(rgb2gray(img))
	segments_watershed = watershed(gradient, markers=50, compactness=0.01)
	seg_map = segments_watershed
	return seg_map

def get_heated_block(seg_map, heat_map):
	label_color = [255, 255, 255]
	chromastism = 6
	min_pix = 20
	min_ratio = 0.4

	[H, W] = seg_map.shape
	num_seg = max(seg_map.flatten())
	count_seg = [0 for x in xrange(num_seg)]
	pix_seg = [0 for x in xrange(num_seg)]
	seg_select = []

	for i in xrange(H):
		for j in xrange(W):
			seg_idx = seg_map[i, j]
			pix_seg[seg_idx - 1] += 1
			if np.all(np.abs(heat_map[i, j, :] - label_color) < chromastism):
				count_seg[seg_idx - 1] += 1

	for idx in xrange(num_seg):
		if count_seg[idx] >= min_pix or (float(count_seg[idx]) / pix_seg[idx]) >= min_ratio:
			seg_select.append(idx + 1)
	return seg_select

def block_spread(img, seg_map, seg_idx):
	pass

def fill_image(img, seg_map, seg_idx):
	label_color = [255, 255, 255]
	img[np.isin(seg_map, seg_idx), :] = label_color
	return img

def refine_without_block(img, img_guide):
	label_color = [255, 255, 255]
	target_color = [255, 255, 255]
	chromastism = 6
	H, W, _ = img.shape
	for i in xrange(H):
		for j in xrange(W):
			if np.all(np.abs(img[i, j, :] - label_color) < chromastism):
				img[i, j, :] = target_color
	return img

def main():
	input_folder = 'data/input/occluded_body_images'
	guide_folder = 'data/output/voc_full_4w'
	output_folder = 'data/test'

	img_name = '035_004.jpg'
	input_path = os.path.join(input_folder, img_name)
	guide_path = os.path.join(guide_folder, img_name)
	output_path = os.path.join(output_folder, img_name)
	
	img = np.array(Image.open(input_path))
	img_guide = np.array(Image.open(guide_path))

	path_guide = os.path.join(output_folder, 'guide_' + img_name)
	Image.fromarray(img_guide).save(path_guide)

	path_simple = os.path.join(output_folder, 'simple_' + img_name)
	img_simple = refine_without_block(img, img_guide)
	Image.fromarray(img_simple).save(path_simple)

	seg_map = segment_image(img)
	print(seg_map, min(seg_map.flatten()), max(seg_map.flatten()))
	[H, W] = seg_map.shape
	heated_block = get_heated_block(seg_map, img_guide)
	print(heated_block)
	output_img = fill_image(img, seg_map, heated_block)
	Image.fromarray(output_img).save(output_path)

if __name__ == '__main__':
	main()

