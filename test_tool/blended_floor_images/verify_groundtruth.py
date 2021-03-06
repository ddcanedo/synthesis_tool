import cv2
import os
import math
import numpy as np


gt_path = 'groundtruth/'
img_path = 'images/train/'


for gt in (os.listdir(gt_path)):

	mask = cv2.imread(gt_path + gt,0)
	og = cv2.imread(img_path + gt)

	print(gt)

	contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	cv2.drawContours(og, contours, -1, (0,255,0), 3)

	cv2.namedWindow('img', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('img', 1440, 1080) 
	cv2.imshow('img', og)
	cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('mask', 1440, 1080) 
	cv2.imshow('mask', mask)
	cv2.waitKey()