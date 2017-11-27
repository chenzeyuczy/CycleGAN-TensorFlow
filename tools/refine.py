#! /usr/bin/env python

# Script to refine input image with detection result from cycleGAN.

import numpy as np
from PIL import Image
from scipy import misc
import os

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.util import img_as_float

def segment_image(img):
	img = img_as_float(img)
	gradient = sobel(rgb2gray(img))
	segments_watershed = watershed(gradient, markers=50, compactness=0.01)
	seg_map = segments_watershed
	return seg_map

def get_heated_block(seg_map, heat_map):
	label_color = [255, 255, 255]
	chromastism = 5
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

def block_spread(img, seg_map, seg_seed):
	# Construct adjacent graph.
	num_seg = np.max(seg_map)
	H, W = seg_map.shape
	adjacent_nodes = [set() for x in xrange(num_seg)]
	neighbour = {}
	neighbour['4'] = [[-1, 0], [0, -1], [1, 0], [0, 1]]
	neighbour['8'] = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

	neighbour_type = '4'
	for i in xrange(H):
		for j in xrange(W):
			seg_idx = seg_map[i, j]
			for offset in neighbour[neighbour_type]:
				nei_y, nei_x = i + offset[0], j + offset[1]
				if nei_y < 0 or nei_y >= H or nei_x < 0 or nei_x >= W:
					continue
				adjacent_nodes[seg_idx - 1].add(seg_map[nei_y, nei_x])

	# Calculate average color feature in each block.
	avg_color = [None for x in xrange(num_seg)]
	for i in xrange(num_seg):
		adjacent_nodes[i].remove(i + 1)
		seg_pix = np.vstack(np.where(seg_map == (i + 1)))
		avg_color[i] = np.mean(img[seg_pix[0], seg_pix[1], :], axis=0)

	print(seg_seed, type(seg_seed))
	seg_final = set(seg_seed)
	seg_choose = set(seg_seed)

	# Caution: It is quite hard to determine the threshold!
	# Append adjacent blocks with similar apprearance with seed block into selection.
	threshold = 12
	for root_seg in seg_seed:
		seg_choose = set([root_seg])
		while True:
			seg_new = set()
			for child_seg in seg_choose:
				for leaf_seg in adjacent_nodes[child_seg - 1]:
					if leaf_seg in seg_new or leaf_seg in seg_choose:
						continue
					if np.linalg.norm(avg_color[leaf_seg - 1] - avg_color[child_seg - 1]) <= 		threshold:
						seg_new.add(leaf_seg)
			for child_seg in seg_new:
				seg_choose.add(child_seg)
			if len(seg_new) == 0:
				break
		for item in seg_choose:
			seg_final.add(item)
	return sorted(list(seg_final))

def fill_image(img, seg_map, seg_idx):
	label_color = [255, 255, 255]
	img[np.isin(seg_map, seg_idx), :] = label_color
	return img

def refine(input_img, guide_img):
	seg_map = segment_image(input_img)
	heated_block = get_heated_block(seg_map, guide_img)
	output_img = fill_image(input_img, seg_map, heated_block)
	return output_img

def main():
	input_dir = 'data/input/occluded_body_images'
	guide_dir = 'data/output/pure_3w'
	refine_dir = 'data/refine/pure_3w'

	input_files = os.listdir(input_dir)
	guide_files = os.listdir(guide_dir)
	if not os.path.exists(refine_dir):
		os.makedirs(refine_dir)
	for filename in input_files:
		if filename not in guide_files:
			raise '{} not found in guide directory.'.format(filename)
		input_file = os.path.join(input_dir, filename)
		guide_file = os.path.join(guide_dir, filename)
		output_file = os.path.join(refine_dir, filename)

		input_img = np.array(Image.open(input_file))
		guide_img = np.array(Image.open(guide_file))
		output_img = refine(input_img, guide_img)
		Image.fromarray(output_img).save(output_file)

#	input_path = 'data/input/035_004.jpg'
#	guide_path = 'data/output/035_004.jpg'
#	output_path = 'data/test_refine.jpg'
#
#	input_img = np.array(Image.open(input_path))
#	guide_img = np.array(Image.open(guide_path))
#
#	output_img = refine(input_img, guide_img)
#	img = Image.fromarray(output_img)
#	img.save(output_path)
	pass

if __name__ == '__main__':
	main()

