# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:06:18 2017

@author: rmb-jx
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt 
from skimage.filters import sobel
from skimage.viewer import ImageViewer
from skimage.morphology import watershed
from scipy import ndimage as ndi
from skimage import data
from skimage.segmentation import mark_boundaries

#coins = data.coins()
img = cv2.imread('../data/pen.jpg')
coins = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2

elevation_map = sobel(coins)
segmentation = watershed(elevation_map, markers)

segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_coins, _ = ndi.label(segmentation)
marked = mark_boundaries(coins, labeled_coins,color=(1, 0, 0))

viewer = ImageViewer(marked)
viewer.show()