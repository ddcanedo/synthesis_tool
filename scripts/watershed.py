# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:15:14 2017

@author: rmb-jx
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt 
from skimage.filters import sobel
from skimage.viewer import ImageViewer
from skimage.morphology import watershed
from scipy import ndimage as ndi

img = cv2.imread('../data/pen.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
test = thresh.copy()

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)     # remove out noise
closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 3)   # remove inner noise

 # sure background area                                      #background is small as actual
sure_bg = cv2.dilate(closing,kernel,iterations=3)
# Finding sure foreground area
#dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)       # optional
ret, sure_fg = cv2.threshold(closing,0.9*opening.max(),255,0)       # erosion, foreground is smaller as actual
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

#contours, hierarchy = cv2.findContours(test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#img1 = cv2.drawContours(img, contours, 0, (0,255,0), 3)
'''
mark = gray.copy()
mark[unknown != 0] = 1
mark[sure_fg != 0] = 2
mark = np.uint32(mark)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg) 
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1 
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
'''

contours, hierarchy = cv2.findContours(sure_fg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      
#Creating a numpy array for markers and converting the image to 32 bit using dtype paramter
marker = np.zeros((gray.shape[0], gray.shape[1]),dtype = np.int32)

marker = np.int32(sure_fg) + np.int32(sure_bg)

for id in range(len(contours)):
    cv2.drawContours(marker,contours,id,id+2, -1)

    marker = marker + 1
    marker[unknown==255] = 0

'''
grad_x = np.zeros_like(gray)
grad_y = np.zeros_like(gray)
cv2.sobel(gray,grad_x,  )

cv2.watershed(img, marker)
img[marker==-1]=(0,0,0) # to draw the contours in red 

cv2.imshow('result', gray)
cv2.waitKey(0)
'''

elevation_map = sobel(gray)
segmentation = watershed(elevation_map, marker)
labeled_coins, _ = ndi.label(segmentation)

viewer = ImageViewer(segmentation)
viewer.show()


