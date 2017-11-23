#! /usr/bin/env python

import numpy as np
from PIL import Image

def evaluate(img, bbx, color):
	[x, y, w, h] = bbx
	[H, W, _] = img.shape
	print(x, y, w, h)
	print(H, W, _)

	chromatism = 3

	# img_gt = np.zeros((H, W), dtype=np.uint8)
	# img_gt[y: y + h, x: x + w] = 255
	# Image.fromarray(img_gt).save('data/gt.jpg')

	# roi = np.where(np.all(img == color, axis=2))
	# roi = zip(roi[0], roi[1])
	roi = [[] for i in range(2)]
	for i in xrange(H):
		for j in xrange(W):
			if np.all(np.abs(img[i, j, :] - color) < chromatism):
				roi[0].append(i)
				roi[1].append(j)
	roi = np.array(roi)

	img_detect = np.zeros((H, W), dtype=np.uint8)
	# for pos in roi:
	# 	img_detect[pos[0], pos[1]] = 255
	img_detect[roi[0], roi[1]] = 255
	print img_detect
	# print(roi.shape)
	# for i in range(roi.shape[1]):
	# 	img_detect[roi[:,i]] = 255
	# img_detect[roi] = 255
	
	return img_detect

def main():
	img_path = '/home/zeyu/data/MARS/noise_multipatch/white/0005C5T0113F058.jpg'
	export_file = 'data/test.jpg'
	bbx = [16, 134, 78, 107]
	color = (255, 255, 255)
	img = Image.open(img_path)
	img.save('data/original.jpg')
	img = np.array(img)
	detect = evaluate(img, bbx, color)
	print detect, detect.shape
	img_export = Image.fromarray(detect)
	img_export.save(export_file)

if __name__  == '__main__':
	main()
