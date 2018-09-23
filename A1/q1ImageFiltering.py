#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2

img = cv2.imread('lena.tif')
s = 7

f = cv2.getGaussianKernel(s*4+1, -1)
#an array of s*4+1 by 1 coefficients
#get dot product to convert to square matrix
#print f, f.T
f = np.dot(f, f.T)
print f

def my_imfilter(img, f):
    rows, cols, channels = img.shape
    print f.shape
    forder, temp = f.shape
    #check if filter matrix is even
    if forder%2 == 0:
        print "Filter matrix order is even: cannot blur. Must be odd."
        return

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padded_grey_img = cv2.copyMakeBorder(grey_img, forder/2, forder/2, forder/2, forder/2, cv2.BORDER_CONSTANT, 0)    
    out_img = np.zeros([rows, cols, channels], dtype=np.uint8)

    #print padded_grey_img[0:0+forder, 0:0+forder]
    
    for y in range(rows):
        for x in range(cols):
            #snapshot = np.array((padded_grey_img[y][x:x+3], padded_grey_img[y+1][x:x+3], padded_grey_img[y+2][x:x+3]))
            weighted = np.multiply(f, padded_grey_img[y:y+forder, x:x+forder])
            pixel = np.sum(weighted)
            out_img[y][x] = pixel
            #print out_img[y][x][0], grey_img[y][x]

    cv2.imshow('grey_lena', grey_img)
    cv2.imshow('blur_lena', out_img)
    cv2.waitKey()
    return out_img

my_imfilter(img, f)

