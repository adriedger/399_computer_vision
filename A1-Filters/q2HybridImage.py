#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2
import q1ImageFiltering as q1

def create_hybrid_image(img1, img2, lowfq, highfq):
    lpf = cv2.getGaussianKernel(lowfq, -1)
    low_pass = q1.my_imfilter(img1, lpf)

    lpf2 = cv2.getGaussianKernel(highfq, -1)
    low_pass2 = q1.my_imfilter(img2, lpf2)
    high_pass = np.subtract(img2, low_pass2[:,:,0])

    hybrid = np.add(low_pass[:,:,0], high_pass)

    return low_pass, high_pass, hybrid

def test():
    img1 = cv2.imread('data/dog.bmp')
    img2 = cv2.imread('data/cat.bmp')
    lowfreq = 99
    highfreq = 35
    grey_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    blur, sharp, hybrid = create_hybrid_image(grey_img1, grey_img2, lowfreq, highfreq)

    pG = hybrid.copy()
    G = hybrid.copy()
    for i in range(4):
        G = cv2.pyrDown(G)
        pad = np.ones((hybrid.shape[0]-G.shape[0],G.shape[1]))
        tmp = np.vstack((pad, G))
        pG = np.hstack((pG,tmp))
    
    cv2.imshow('lp', blur)
    cv2.imshow('hp', sharp)
    cv2.imshow('hybrid', hybrid)
    cv2.imshow('img', pG.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('out_data/lp.tif', blur)
    cv2.imwrite('out_data/hp.tif', sharp)
    cv2.imwrite('out_data/hybrid.tif', hybrid)
    cv2.imwrite('out_data/image_pyramid.tif', pG)

