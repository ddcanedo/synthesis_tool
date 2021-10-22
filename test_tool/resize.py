import os
import cv2
import numpy as np

path = 'patterns/'
save = 'resized_patterns/'

images = []
for file in sorted(os.listdir(path)):
    images.append(file)

for i in range(0, len(images)):
	if len(images[i].split('_')) > 1:
		print(images[i])
		img = cv2.imread(path+images[i],0)
		img = cv2.resize(img, (640,640))
		cv2.imwrite(save + images[i], img)
	else:
		img = cv2.imread(path+images[i])
		img = cv2.resize(img, (640,640))
		cv2.imwrite(save + images[i], img)

