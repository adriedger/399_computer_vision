#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2 

#img = cv2.imread('lena.tif')
img = cv2.imread('data_q2/checkboard.png')
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
    
    #get II's, run a gaussian filter through them
    IxIx = cv2.GaussianBlur(Ix*Ix, (3, 3), 0)
    IxIy = cv2.GaussianBlur(Ix*Iy, (3, 3), 0)
    IyIy = cv2.GaussianBlur(Iy*Iy, (3, 3), 0)
    #print np.amax(IxIx), np.amin(IxIx)
    #print np.amax(IxIy), np.amin(IxIy)
    #print np.amax(IyIy), np.amin(IyIy)
        
    #need to be float to save pixels as correct data type
    c_img = np.zeros([row, col, 1], dtype=np.float32)
    out_img = np.zeros([row, col, 1], dtype=np.uint8)
    
    #calculate c for each pixel, threshold
    for y in range(3, row-3):
        for x in range(3, col-3):
            
            m1 = np.sum(IxIx[y:y+3, x:x+3])
            m2 = np.sum(IxIy[y:y+3, x:x+3])
            m3 = np.sum(IyIy[y:y+3, x:x+3])
            c = (m1*m3-m2*m2) - 0.04*(m1+m3)
            #threshold
            if c > 10000000000:
                c_img[y, x] = c
                #out_img[y, x] = 255
            else:
                c_img[y, x] = 0
                #out_img[y, x] = 0
    
    #take max of adjacent c pixels, supress lower values
    for y in range(0, row, 3):
        for x in range(0, col, 3):
            
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(c_img[y:y+3, x:x+3]) 
            #keep highest, supress lower, Loc returns as a Point(x, y)
            if maxVal > 0:
                #print minVal, maxVal, minLoc, maxLoc
                locX, locY = maxLoc
                out_img[y+locY, x+locX] = 255
    
    cv2.imshow('cornerness', out_img)
    cv2.waitKey()

my_harris(img, 0, 9)
