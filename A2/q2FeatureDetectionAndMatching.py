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
    #print row, col
    #calculate derivatives as float32(so can include the negative values)
    Ix = cv2.Sobel(grey_img, cv2.CV_32F, 1, 0, 3)
    Iy = cv2.Sobel(grey_img, cv2.CV_32F, 0, 1, 3)
    #print np.amax(abs(Ix))
    #between -1, 1, will help visualize
    #Ix = (Ix / np.amax(abs(Ix)))    
    #Iy = (Iy / np.amax(abs(Iy)))    
    #print np.amax(Ix), np.amin(Ix)
    IxIx = Ix*Ix
    IxIy = Ix*Iy
    IyIy = Iy*Iy
    print np.amax(IxIx), np.amin(IxIx)
    print np.amax(IxIy), np.amin(IxIy)
    print np.amax(IyIy), np.amin(IyIy)
    #IxIx_norm = cv2.normalize(IxIx, 0.0, 1.0, cv2.NORM_MINMAX)
    #IxIy_norm = cv2.normalize(IxIy, 0.0, 1.0, cv2.NORM_MINMAX)
    #IyIy_norm = cv2.normalize(IyIy, 0.0, 1.0, cv2.NORM_MINMAX)
    #print np.amax(IxIx_norm), np.amin(IxIx_norm)
    #print np.amax(IxIy_norm), np.amin(IxIy_norm)
    #print np.amax(IyIy_norm), np.amin(IyIy_norm)

    #cv2.imshow('IxIx', IxIx)
    #cv2.imshow('IxIy', IxIy)
    #cv2.imshow('IyIy', IyIy)
    #need to be float to save pixels as correct data type
    out_img = np.zeros([row, col, 1], dtype=np.float32)

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
            out_img[y, x] = c
    print np.amax(out_img), np.amin(out_img)
    out_img = (out_img/np.amax(abs(out_img)))*255
    print np.amax(out_img), np.amin(out_img)
    out_img = cv2.convertScaleAbs(out_img)
    #print np.amax(out_img), np.amin(out_img)
    cv2.imshow('cornerness', out_img*1000)
    cv2.waitKey()

my_harris(img, 0, 9)
