import os
import cv2
import numpy as np

def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())
path = 'floors/patterns/'

images = []
for file in sorted(os.listdir(path)):
    images.append(file)

dupes = []
for i in range(0, len(images)):
    print(images[i])
    img = cv2.imread(path+images[i])
    for j in range(i+1, len(images)):
        aux = cv2.imread(path+images[j])
        if is_similar(img, aux):
            dupes.append(images[j])
    print(dupes)

print(dupes)