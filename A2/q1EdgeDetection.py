#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2 


#k = np.array([[1, 0], [0, -1]])
#a = np.array([[1],[-1]])
#dot = np.multiply(a, a.T)
#print k
#print k.T
#print a
#print a.T
#print dot

#img = cv2.imread('data_q1/bsds_3096.jpg')
img = cv2.imread('lena.tif')
cv2.imshow('input', img)

def edge_detection(img, method):

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == 'diff':
        diff_x = cv2.filter2D(grey_img, -1, np.array([[0, 1], [-1, 0]]))
        diff_y = cv2.filter2D(grey_img, -1, np.array([[1, 0], [0, -1]]))

        #cv2.imshow('diff_x', diff_x)
        #cv2.imshow('diff_y', diff_y)

        #gradient_mag = diff_x^2 + diff_y^2
        #convert to CV_32 or 64
        #gradient_mag = cv2.sqrt(gradient_mag)
        #convert back to original depth
        #diff_x_abs = cv2.convertScaleAbs(diff_x)
        #diff_y_abs = cv2.convertScaleAbs(diff_y)

        diff_x = np.float32(diff_x)
        diff_y = np.float32(diff_y)

        gradient_mag = cv2.magnitude(diff_x, diff_y)

        gradient_mag = cv2.convertScaleAbs(gradient_mag)

        return gradient_mag
    
    if method == 'sobel':
        grad_x = cv2.Sobel(grey_img, -1, 1, 0, 3)
        grad_y = cv2.Sobel(grey_img,-1, 0, 1, 3)

        gradient = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        
        return gradient

    if method == 'laplacian':
        gradient = cv2.Laplacian(grey_img, -1, 3)
        
        return gradient


diff = edge_detection(img, 'diff')
#sobel = edge_detection(img, 'sobel')
#laplacian = edge_detection(img, 'laplacian')

cv2.imshow('diff', diff)
#cv2.imshow('sobel', sobel)
#cv2.imshow('laplacian', laplacian)
cv2.waitKey()
