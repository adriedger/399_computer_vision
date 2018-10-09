#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2 


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
    c_img = np.zeros([row, col], dtype=np.float32)
    
    #calculate c for each pixel
    for y in range(row-blocksize):
        for x in range(col-blocksize):
            
            m1 = np.sum(IxIx[y:y+blocksize, x:x+blocksize])
            m2 = np.sum(IxIy[y:y+blocksize, x:x+blocksize])
            m3 = np.sum(IyIy[y:y+blocksize, x:x+blocksize])
            c = (m1*m3-m2*m2) - k*(m1+m3)
            #assign cornerness to middle pixel
            mid = blocksize//2
            c_img[y+mid, x+mid] = c
    
    #keep the max of blocksize of c_img, supress lower values
    for y in range(0, row, blocksize):
        for x in range(0, col, blocksize):
            
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(c_img[y:y+blocksize, x:x+blocksize]) 
            #keep highest, Loc returns as a Point(x, y)
            locX, locY = maxLoc
            c_img[y:y+blocksize, x:x+blocksize] = 0
            c_img[y+locY, x+locX] = maxVal

    return c_img
    
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

    cv2.imshow('input', img)

    out1 = my_harris(img, 0.04, 2)
    
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out2 = cv2.cornerHarris(grey_img, 2, 3, 0.04)
    
    #threshold, visualize
    img[out1>0.01*out1.max()]=[0,0,255]
    img_copy[out2>0.01*out2.max()]=[0,0,255]

    cv2.imshow('my_harris', img)
    cv2.imshow('cv2_cornerHarris', img_copy)
    cv2.waitKey()

def test2_2():
    #0 on imread returns a greyscale
    img = cv2.imread('data_q2/episcopal_gaudi1.jpg',0)  
    img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #img = cv2.imread('lena.tif', 0)  
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    
    # visualize results
    img_d = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('SIFTfeatures', img_d)
    cv2.imshow('sift_keypoints', des)
    cv2.waitKey(0)
    #cv2.imwrite('sift_keypoints.jpg',img)


test2_1()
test2_2()
