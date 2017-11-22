#! /usr/bin/env python

import numpy as np
from PIL import Image

def evaluate(img_file, bbx, color):
	img = np.array(Image.open(img_file))
	[x, y, w, h] = bbx
	[H, W, ~] = img.shape
	roi = np.where(np.all(img == color, axis=-1))
	roi = np.vstack(roi)
	img_roi = np.zeros((H, W), dtype=int)
	img_roi[roi] = 1
	img_gt = np.zeros((H, W), dtype=int)
	img_gt[x: x + w, y: y + h] = 1
	img_detect = np.zeros((H, W), dtype=int)
	img_detect[roi] = 1
	print(roi)

def main():
	img = '/home/zeyu/data/MARS/noise/white/0001C2T0050F013.jpg'
	bbx = [16, 134, 78, 107]
	color = (255, 255, 255)
	evaluate(img, bbx, color)

