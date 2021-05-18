import cv2
import os
import math
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
	for line in f:
		x = float(line.split(' ')[1])
		y = float(line.split(' ')[2])
		w = float(line.split(' ')[3])
		h = float(line.split(' ')[4])

		tl = (round((x-w/2)*img.shape[1]), round((y-h/2)*img.shape[0]))
		br = (round((x+w/2)*img.shape[1]), round((y+h/2)*img.shape[0]))


		cv2.rectangle(img, tl, br, (0,0,255), 2)

		

	f.close()

	cv2.namedWindow('img', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('img', 1440, 1080) 
	cv2.imshow('img', img)
	cv2.waitKey()



