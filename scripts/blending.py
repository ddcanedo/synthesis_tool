# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:51:18 2017

@author: rmb-jx
"""

from skimage.io import imread, imsave
from skimage.viewer import ImageViewer
from skimage.morphology import erosion, dilation
import numpy as np

clean_ground = imread('../build/ground_texture.jpg')
arti_dirt = imread('../data/arti_dirt.jpg')
boundary_mask = imread('../data/mask/boundry_mask.bmp')
segment_mask = imread('../data/mask/segmentation_mask.bmp')

boundary_mask_3c = np.repeat(boundary_mask[:, :, np.newaxis], 3, axis=2)  
segment_mask_3c = np.repeat(segment_mask[:, :, np.newaxis], 3, axis=2)     # 0 for object
segment_mask_3c = dilation(segment_mask_3c )   ## 0 for object, to make the dilated object smaller as 'original'
                                               ## size, to fit the size of the artificial dirt
blending = np.multiply(clean_ground ,segment_mask_3c)
imsave('../data/mask/blending1.bmp',blending)
#temp = np.multiply(arti_dirt, (1 - segment_mask_3c))
temp = arti_dirt.copy()
temp[segment_mask_3c == 1] = 0
blending = blending + temp

viewer = ImageViewer(blending)        # visulization, for debugging
viewer.show()

imsave('../data/mask/blending2.bmp',blending)
imsave('../data/mask/blending3.bmp',temp) 