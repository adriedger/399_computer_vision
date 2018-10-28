#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2 

#img = cv2.imread('data_q1/bsds_3096.jpg')
img = cv2.imread('A2/lena.tif', 0)
cv2.imshow('in', img)

#pixel = img[:, :]
img[img[:, :]<100] = 0

cv2.imshow('out', img)
cv2.waitKey()
