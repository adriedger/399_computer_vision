#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2 

#img = cv2.imread('data_q1/bsds_3096.jpg')
img = cv2.imread('lena.tif')
cv2.imshow('input', img)

def edge_detection(img, method):

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == 'sobel':
        grad_x = cv2.Sobel(grey_img, -1, 1, 0, 3)
        grad_y = cv2.Sobel(grey_img,-1, 0, 1, 3)

        gradient = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        
        return gradient

    if method == 'laplacian':
        gradient = cv2.Laplacian(grey_img, -1, 3)
        
        return gradient


sobel = edge_detection(img, 'sobel')
laplacian = edge_detection(img, 'laplacian')

cv2.imshow('sobel', sobel)
cv2.imshow('laplacian', laplacian)
cv2.waitKey()
