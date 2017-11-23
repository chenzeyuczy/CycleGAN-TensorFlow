#! /usr/bin/env python

# Script to refine input image with detection result from cycleGAN.

import numpy as np
from PIL import Image
from scipy import misc

def get_ocllusion(img, color, chromatism=5):
	H, W, _ = img.shape
	roi = [[] for i in range(2)]
	for i in xrange(H):
		for j in xrange(W):
			if np.all(np.abs(img[i, j, :] - color) < chromatism):
				roi[0].append(i)
				roi[1].append(j)
	roi = np.array(roi)

	return roi

def fill_occlusion(img, region, color):
	img[region[0], region[1], :] = color
	return

def refine(in_img, guide_img):
	label_color = (255, 255, 255)
	refine_color = (255, 255, 255)

	guide_img = misc.imresize(guide_img, in_img.shape)
	occluded_region = get_ocllusion(guide_img, label_color, chromatism=3)
	out_img = in_img.copy()
	fill_occlusion(out_img, occluded_region, refine_color)
	return out_img

def main():
	input_path = 'data/input/035_004.jpg'
	guide_path = 'data/output/035_004.jpg'
	output_path = 'data/test_refine.jpg'

	input_img = np.array(Image.open(input_path))
	guide_img = np.array(Image.open(guide_path))

	output_img = refine(input_img, guide_img)
	img = Image.fromarray(output_img)
	img.save(output_path)
	pass

if __name__ == '__main__':
	main()

