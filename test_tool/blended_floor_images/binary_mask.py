import cv2
import os
import math
import numpy as np
labels_path = 'labels/train/'
img_path = 'images/train/'

labels = []

for file in (os.listdir(labels_path)):
	if file.split('.')[1] == 'txt':
		labels.append(labels_path + file)


print(labels)

for label in labels:
	l = label
	aux = label.split('/')[2]
	img = img_path + aux.split('.')[0] + '.png'

	print(img)


	f = open(l, "r")
	img = cv2.imread(img)
	mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

	for line in f:
		x = float(line.split(' ')[1])
		y = float(line.split(' ')[2])
		w = float(line.split(' ')[3])
		h = float(line.split(' ')[4])

		tl = (round((x-w/2)*img.shape[1]), round((y-h/2)*img.shape[0]))
		br = (round((x+w/2)*img.shape[1]), round((y+h/2)*img.shape[0]))


		mask[tl[1]:br[1], tl[0]:br[0]] = 255


	f.close()

	cv2.imwrite('groundtruth/' + aux.split('.')[0] + '.png', mask)