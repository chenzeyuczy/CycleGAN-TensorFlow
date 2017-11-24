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
	gradient = sobel(rgb2gray(img))
	segments_watershed = watershed(gradient, markers=50, compactness=0.01)
	img_seg = segments_watershed
	return img_seg

def main():
	input_folder = '/home/zeyu/data/Partial-REID_Dataset/occluded_body_images'
	output_folder = 'data/segment'
	
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	img_files = os.listdir(input_folder)
	for img_file in img_files:
		input_path = os.path.join(input_folder, img_file)
		output_path = os.path.join(output_folder, img_file)
		img = np.array(Image.open(input_path))
		img = img_as_float(img)
		img_segment = segment_image(img)
		[H, W] = img_segment.shape
		output_img = mark_boundaries(img, img_segment)
		Image.fromarray((output_img * 255).astype(np.uint8)).save(output_path)

if __name__ == '__main__':
	main()

