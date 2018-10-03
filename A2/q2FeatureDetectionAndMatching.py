#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision
#corners are where m1,m3 are high and m2 is 0

import numpy as np
import cv2 

img = cv2.imread('lena.tif')
cv2.imshow('input', img)

def my_harris(img, k, blocksize):

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row, col = grey_img.shape
  
    Ix = cv2.filter2D(grey_img, cv2.CV_32F, np.array([[0, 1], [-1, 0]]))
    Iy = cv2.filter2D(grey_img, cv2.CV_32F, np.array([[1, 0], [0, -1]]))
    IxIx = Ix*Ix
    IxIy = Ix*Iy
    IyIy = Iy*Iy

    out_img = np.zeros([row, col, 1], dtype=np.uint8)

    for y in range(3, row-3):
        for x in range(3, col-3):
            #print(IxIy[y:y+3, x:x+3])
            m1 = np.sum(IxIx[y:y+3, x:x+3])
            m2 = np.sum(IxIy[y:y+3, x:x+3])
            m3 = np.sum(IyIy[y:y+3, x:x+3])
            #print(m1, m2, m3)
            c = (m1*m3-m2*m2) - 0.04*(m1+m3)
            print(c)
            #out_img[x, y] = cv2.convertScaleAbs(c)

    cv2.imshow('cornerness', out_img)
    cv2.waitKey()

my_harris(img, 0, 9)
