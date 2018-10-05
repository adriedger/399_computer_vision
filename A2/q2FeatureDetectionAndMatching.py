#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision
#corners are where m1,m3 are high and m2 is 0

import numpy as np
import cv2 

img = cv2.imread('lena.tif')
#img = cv2.imread('data_q2/checkboard.png')
cv2.imshow('input', img)

def my_harris(img, k, blocksize):

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row, col = grey_img.shape
    
    #calculate derivatives as float32(so can include the negative values)
    Ix = cv2.Sobel(grey_img, cv2.CV_32F, 1, 0, 3)
    Iy = cv2.Sobel(grey_img, cv2.CV_32F, 0, 1, 3)
    
    #between -1, 1, will help visualize
    #Ix = (Ix / np.amax(abs(Ix)))    
    #Iy = (Iy / np.amax(abs(Iy)))

    IxIx = Ix*Ix
    IxIy = Ix*Iy
    IyIy = Iy*Iy
    print np.amax(IxIx), np.amin(IxIx)
    print np.amax(IxIy), np.amin(IxIy)
    print np.amax(IyIy), np.amin(IyIy)
        
    #need to be float to save pixels as correct data type
    out_img = np.zeros([row, col, 3], dtype=np.uint8)

    for y in range(3, row-3):
        for x in range(3, col-3):
            #print Ix[y,x]
            #print(IxIy[y:y+3, x:x+3])
            m1 = np.sum(IxIx[y:y+3, x:x+3])
            m2 = np.sum(IxIy[y:y+3, x:x+3])
            m3 = np.sum(IyIy[y:y+3, x:x+3])
            c = (m1*m3-m2*m2) - 0.04*(m1+m3)
            #if c > 0.5:
                #print x, y, m1, m2, m3, c
            if c > 100000000000:    
                out_img[y, x] = [0, 0, 255]
            else:
                out_img[y, x] = [0, 0, 0]
    
    cv2.imshow('cornerness', out_img)
    cv2.waitKey()

my_harris(img, 0, 9)
