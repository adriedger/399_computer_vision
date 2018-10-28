#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

from __future__ import print_function
import numpy as np
import cv2 
import time, os

# computes point on a grid of block_size for an image of size [img_width,img_height]**Dana's code
def getGridPoints(img_width,img_height,block_size):
    _X = np.asarray(range(0, img_width - 1, block_size)) + block_size // 2
    _Y = np.asarray(range(0, img_height - 1, block_size)) + block_size // 2

    [X, Y] = np.meshgrid(_X, _Y)
    return X,Y




#cv2.waitKey()
