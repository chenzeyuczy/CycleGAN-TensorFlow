from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

def on_click(event, img, seg_map, selections):
	if not event.xdata or not event.ydata:
		return

	selection_occlusion, selection_person = selections
	x, y = int(event.xdata), int(event.ydata)
	seg_label = seg_map[y, x]
	if event.button == 1:  # Press left button.
		if seg_label not in selection_occlusion:
			selection_occlusion.add(seg_label)
			selection_person.discard(seg_label)
		else:
			selection_occlusion.discard(seg_label)
	elif event.button == 3:  # Press right button.
		if seg_label not in selection_person:
			selection_person.add(seg_label)
			selection_occlusion.discard(seg_label)
		else:
			selection_person.discard(seg_label)
	else:
		return

	img_copy = img.copy()
	for label in selection_occlusion:
		position = np.where(seg_map == label)
		position = np.vstack(position)
		img_copy[position[0], position[1], 0] = 255
	for label in selection_person:
		position = np.where(seg_map == label)
		position = np.vstack(position)
		img_copy[position[0], position[1], 1] = 255

	plt.imshow(mark_boundaries(img_copy, seg_map))
	fig.canvas.draw()

	print('You pressed', event.button, event.xdata, event.ydata)

def on_key(event, img, seg_map, selections, output_path):
	if event.key == 'enter':  # Press enter to save result.
		h, w, _ = img.shape
		img_output = np.zeros((h, w), dtype=np.uint8)
		occlusion, person = selections
		for label in selection_occlusion:
			position = np.where(seg_map == label)
			position = np.vstack(position)
			img_output[position[0], position[1]] = 255
		for label in selection_person:
			position = np.where(seg_map == label)
			position = np.vstack(position)
			img_output[position[0], position[1]] = 127
		Image.fromarray(img_output).save(output_path)
		plt.close(fig)
	if event.key == 'c':  # Press escape to discard all selection.
		for selection in selections:
			selection.clear()
		plt.imshow(mark_boundaries(img, seg_map))
		fig.canvas.draw()
	print(type(event.key))
	print('You pressed', event.key, event.xdata, event.ydata)


input_root = '/home/zeyu/data/Partial-REID_Dataset/occluded_body_images'
output_root = '/home/zeyu/data/Partial-REID_Dataset/occlusion_label'

idx_start, idx_end = 0, -1

if not os.path.exists(output_root):
	os.makedirs(output_root)

filenames = sorted(os.listdir(input_root))
for filename in filenames[idx_start: idx_end]:
	input_path = os.path.join(input_root, filename)
	output_path = os.path.join(output_root, filename)
	img = np.array(Image.open(input_path))
	img_label = img.copy()

	gradient = sobel(rgb2gray(img_as_float(img)))
	segments_watershed = watershed(gradient, markers=80, compactness=0.01)

	print('Compact watershed number of segments: {}'.format(len(np.unique(segments_watershed))))

	selection_occlusion = set()
	selection_person = set()
	selections = [selection_occlusion, selection_person]

# Bind interactive event.
	fig = plt.figure()
	plt.title(filename)
	fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, img, segments_watershed, selections))
	fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, img, segments_watershed, selections, output_path))

	imgplt = plt.imshow(mark_boundaries(img, segments_watershed))
#imgplt.set_title('Compact watershed')

	plt.tight_layout()
	plt.show()
