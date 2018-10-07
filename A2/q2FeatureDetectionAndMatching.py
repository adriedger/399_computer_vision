#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

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
    
    #get II's, run a gaussian filter through them
    IxIx = cv2.GaussianBlur(Ix*Ix, (3, 3), 0)
    IxIy = cv2.GaussianBlur(Ix*Iy, (3, 3), 0)
    IyIy = cv2.GaussianBlur(Iy*Iy, (3, 3), 0)
        
    #need to be float to save pixels as correct data type
    c_img = np.zeros([row, col, 1], dtype=np.float32)
    out_img = np.zeros([row, col, 1], dtype=np.uint8)
    
    #calculate c for each pixel, threshold
    for y in range(row-3):
        for x in range(col-3):
            
            m1 = np.sum(IxIx[y:y+3, x:x+3])
            m2 = np.sum(IxIy[y:y+3, x:x+3])
            m3 = np.sum(IyIy[y:y+3, x:x+3])
            c = (m1*m3-m2*m2) - 0.04*(m1+m3)
            #threshold, assign cornerness to middle pixel
            if c > 10000000000:
                c_img[y+1, x+1] = c
            else:
                c_img[y+1, x+1] = 0
    
    #keep the max of sections of c_img, supress lower values
    for y in range(0, row, 20):
        for x in range(0, col, 20):
            
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(c_img[y:y+20, x:x+20]) 
            #keep highest, Loc returns as a Point(x, y)
            if maxVal > 0:
                locX, locY = maxLoc
                out_img[y+locY, x+locX] = 255
                img[y+locY-3:y+locY+3, x+locX-3:x+locX+3] = [0, 0, 255]
    
    cv2.imshow('cornerness', out_img)
    cv2.imshow('features', img)
    cv2.waitKey()

my_harris(img, 0, 9)
