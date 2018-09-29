#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2 

#img = cv2.imread('data_q1/bsds_3096.jpg')
img = cv2.imread('lena.tif')
cv2.imshow('input', img)

def edge_detection(img, method):

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == 'diff':
        #convert to float
        diff_x = cv2.filter2D(grey_img, cv2.CV_32F, np.array([[0, 1], [-1, 0]]))
        diff_y = cv2.filter2D(grey_img, cv2.CV_32F, np.array([[1, 0], [0, -1]]))
        #calculate gradient magnitude, theta
        gradient_mag, theta = cv2.cartToPolar(diff_x, diff_y)
        #convert back to uint8 (0,255)
        gradient_mag = cv2.convertScaleAbs(gradient_mag)

        return gradient_mag
    
    if method == 'sobel':
        sobel_x = cv2.Sobel(grey_img, cv2.CV_32F, 1, 0, 3)
        sobel_y = cv2.Sobel(grey_img, cv2.CV_32F, 0, 1, 3)

        gradient_mag, theta = cv2.cartToPolar(sobel_x, sobel_y)
        
        gradient_mag = cv2.convertScaleAbs(gradient_mag)
        
        return gradient_mag

    if method == 'laplacian':
        gradient = cv2.Laplacian(grey_img, -1, 3)
        
        return gradient


diff = edge_detection(img, 'diff')
sobel = edge_detection(img, 'sobel')
#laplacian = edge_detection(img, 'laplacian')

cv2.imshow('diff', diff)
cv2.imshow('sobel', sobel)
#cv2.imshow('laplacian', laplacian)
cv2.waitKey()
