#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2 


def my_harris(img, k, blocksize):

    img_copy = np.copy(img)

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
    c_img = np.zeros([row, col], dtype=np.float32)
    out_img = np.zeros([row, col, 1], dtype=np.uint8)
    
    #calculate c for each pixel, threshold
    for y in range(row-blocksize):
        for x in range(col-blocksize):
            
            m1 = np.sum(IxIx[y:y+blocksize, x:x+blocksize])
            m2 = np.sum(IxIy[y:y+blocksize, x:x+blocksize])
            m3 = np.sum(IyIy[y:y+blocksize, x:x+blocksize])
            c = (m1*m3-m2*m2) - k*(m1+m3)
            #threshold, assign cornerness to middle pixel
            mid = blocksize//2
            c_img[y+mid, x+mid] = c
            #if c > 10000000000:
                #c_img[y+mid, x+mid] = c
            #else:
                #c_img[y+mid, x+mid] = 0

    return c_img
    
    #keep the max of sections of c_img, supress lower values
    for y in range(0, row, 20):
        for x in range(0, col, 20):
            
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(c_img[y:y+20, x:x+20]) 
            #keep highest, Loc returns as a Point(x, y)
            if maxVal > 0:
                locX, locY = maxLoc
                #out_img[y+locY, x+locX] = 255
                img[y+locY-3:y+locY+3, x+locX-3:x+locX+3] = [0, 0, 255]
    
    #cv2.imshow('cornerness', out_img)
    cv2.imshow('features', img)
    cv2.waitKey()

    dst = cv2.cornerHarris(grey_img,2,3,0.04)
    print dst.max(), dst.min()
    img_copy[dst>0.01*dst.max()]=[0,0,255]
    cv2.imshow('features2', img_copy)
    cv2.waitKey()

def sift():

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(grey_img, None)
    out_img = cv2.drawKeypoints(grey_img, kp, None)
    cv2.imshow('sift_keypoints', out_img)
    cv2.waitKey()

def test2_1():
    img = cv2.imread('lena.tif')
    img_copy = np.copy(img)

    #img = cv2.imread('data_q2/checkboard.png')
    cv2.imshow('input', img)

    out1 = my_harris(img, 0.04, 2)
    
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out2 = cv2.cornerHarris(grey_img, 2, 3, 0.04)
    
    print out1.shape, out1.max()
    print out2.shape, out2.max()

    img[out1>0.01*out1.max()]=[0,0,255]
    img_copy[out2>0.01*out2.max()]=[0,0,255]

    cv2.imshow('my_harris', img)
    cv2.imshow('cv2_cornerHarris', img_copy)
    cv2.waitKey()

test2_1()
#img = cv2.imread('lena.tif')
#my_harris(img, 0.04, 2)
#dst = cv2.cornerHarris(gray,2,3,0.04)
#dst = my_harris(img, 0.04, 3)
#img[dst>0.01*dst.max()]=[0,0,255]
#cv2.imshow('features', img)
#cv2.waitKey()
#sift()
