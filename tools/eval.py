#! /usr/bin/env python

# Script to refine input image with detection result from cycleGAN.

import numpy as np
from PIL import Image
from scipy import misc
import os

from skimage.color import rgb2gray, rgb2lab
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.util import img_as_float

def segment_image(img):
	img = img_as_float(img)
	gradient = sobel(rgb2gray(img))
	segments_watershed = watershed(gradient, markers=80, compactness=0.01)
	seg_map = segments_watershed
	return seg_map

def get_heated_block(seg_map, heat_map):
	label_color = [127, 127, 127]
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

def block_propagate(img, seg_map, seg_seed):
	# Construct adjacent graph.
	num_seg = np.max(seg_map)
	H, W = seg_map.shape
	adjacent_nodes = [set() for x in xrange(num_seg)]
	neighbour = {}
	neighbour['4'] = [[-1, 0], [0, -1], [1, 0], [0, 1]]
	neighbour['8'] = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
	img_lab = rgb2lab(img)

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
		adjacent_nodes[i].discard(i + 1)
		seg_pix = np.vstack(np.where(seg_map == (i + 1)))
		avg_color[i] = np.mean(img_lab[seg_pix[0], seg_pix[1], :], axis=0)

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

def get_selection(img, seg_map, seg_idx):
# Function to get selection occording to segment indices.
	label_color = 255
	h, w, _ = img.shape
	img_res = np.zeros((h, w), dtype=np.uint8)
	img_res[np.isin(seg_map, seg_idx)] = label_color
	return img_res
	
def get_performance(img1, img2, label1=255, label2=255):
	assert(img1.shape == img2.shape)
	num_pre = np.sum(img1 == label1)
	num_gt = np.sum(img2 == label2)
	num_inter = np.sum(np.logical_and(img1 == label1, img2 == label2))
	num_union = np.sum(np.logical_or(img1 == label1, img2 == label2))

	if num_union > 0:
		iou = 1.0 * num_inter / num_union
	else:
		iou = 0.0
	if num_pre > 0:
		precision = 1.0 * num_inter / num_pre
	else:
		precision = 0.0
	if num_gt > 0:
		recall = 1.0 * num_inter / num_gt
	else:
		recall = 0.0
	return (precision, recall, iou)

def main():
	dataset_name = 'Partial_REID'
	input_dir = 'data/input/' + dataset_name + '/occluded'
	input_dir = 'data/input/' + dataset_name + '/occlusion_label'
	model_prefix = 'imagenet_cuhk01_square'

	idx_begin, idx_end = 1, 12
	label_occlusion, label_person, label_select = 255, 127, 255
	flag_propagate = False
	flag_export = False

	for iter_time in range(idx_begin, idx_end):
		model_name = model_prefix + '_' + str(iter_time) + 'w'
		guide_dir = 'data/output/' + model_name + '/' + dataset_name

		if flag_export:
			output_dir = 'data/refine/' + model_name + '/' + dataset_name
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)

		input_files = os.listdir(input_dir)
		guide_files = os.listdir(guide_dir)
		label_files = os.listdir(label_dir)

		occlusion_precison = []
		occlusion_recall = []
		person_precision = []
		person_recall = []

		for filename in input_files:
			if filename not in guide_files:
				print('{} not found in guide directory.'.format(filename))
				continue
			if filename not in label_files:
				print('{} not found in label directory.'.format(filename))
				continue

			input_file = os.path.join(input_dir, filename)
			guide_file = os.path.join(guide_dir, filename)
			label_file = os.path.join(label_dir, filename)

			input_img = np.array(Image.open(input_file))
			guide_img = np.array(Image.open(guide_file))
			label_img = np.array(Image.open(label_file))

			seg_map = segment_image(input_img)
			heated_block = get_heated_block(seg_map, guide_img)
			if flag_propagate:
				heated_block = block_propagate(input_img, seg_map, heated_block)

			if flag_export:
				img_output = fill_image(input_img, seg_map, heated_block)
				path_output = os.path.join(output_dir, filename)
				Image.fromarray(img_output, path_output)

			img_select = get_selection(input_img, seg_map, heated_block)
			occlusion_rate = get_performance(img_select, label_img, label_select, label_occlusion)
			person_rate = get_performance(img_select, label_img, label_select, label_person)

			occlusion_precison.append(occlusion_rate[0])
			occlusion_recall.append(occlusion_rate[1])
			person_precision.append(person_rate[0])
			person_recall.append(person_rate[1])

		print('Model: %s' % model_name)
		print('Occlusion Precision: %f\t Occlusion Recall: %f' % (np.mean(occlusion_precison), np.mean(occlusion_recall)))
		print('Person Precision: %f\t Person Recall: %f' % (np.mean(person_precision), np.mean(person_recall)))

	pass

if __name__ == '__main__':
	main()

