# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:09:55 2017

@author: rmb-jx
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt 
from skimage import data
from skimage.filters import sobel
from skimage.viewer import ImageViewer
from skimage.morphology import watershed
from skimage.morphology import binary_opening, binary_closing
from skimage.morphology import erosion, dilation
from scipy import ndimage as ndi
from skimage.io import imsave

from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries

from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray


img = cv2.imread('../data/arti_dirt.jpg')
gray = rgb2gray(img)
img_blur = gaussian(img, 0.5)
gray_blur = gaussian(gray)

threshold = threshold_otsu(img_blur)
binary = img_blur > threshold
binary1d = binary[:,:,0]

markers = np.zeros_like(gray)
markers[binary1d > 0] = 2
markers[binary1d == 0] = 1

elevation_map = sobel(gray_blur)
segmentation = watershed(elevation_map, markers)

#segmentation = ndi.binary_fill_holes(segmentation - 1)
segmentation = segmentation - 1
labeled_coins, _ = ndi.label(segmentation)

marked = mark_boundaries(img, labeled_coins,color=(0, 1, 1))    # mark the boundries
boundries = find_boundaries(labeled_coins)                      # find the boundry labels
boundries_dila = dilation(boundries)
#boundries_dila = dilation(boundries_dila)          # twice dialation for better results

segmentation_dila = erosion(segmentation)
#segmentation_dila = erosion(segmentation_dila)      # 1 indicates the background, inversed
labeled_coins_dila, _ = ndi.label(segmentation_dila)
marked_dila = mark_boundaries(img, labeled_coins_dila,color=(0, 1, 1))    


# save the results
# boundary mask
temp = 1 - boundries_dila
imsave('../data/mask/boundry_mask.bmp',boundries_dila)
temp = np.multiply(temp,gray)
viewer = ImageViewer(temp)        # visulization, for debugging
viewer.show()
# segmentation mask
imsave('../data/mask/segmentation_mask.bmp',segmentation_dila)
viewer = ImageViewer(binary1d)        # visulization, for debugging
viewer.show()

# postprocessing of trandformation of the objects
# shif and rotation the ati dirt image
#imsave('../data/mask/segmentation.bmp',segmentation*125)
#imsave('../data/mask/boundary.bmp',boundries*125)

#imsave('../data/mask/marked_dila.bmp',marked_dila)
#imsave('../data/mask/marked.bmp',marked)