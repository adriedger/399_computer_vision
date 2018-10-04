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
        #return as float (as to include negative values)
        diff_x = cv2.filter2D(grey_img, cv2.CV_32F, np.array([[0, 1], [-1, 0]]))
        diff_y = cv2.filter2D(grey_img, cv2.CV_32F, np.array([[1, 0], [0, -1]]))
        #calculate gradient magnitude, theta
        gradient_mag, theta = cv2.cartToPolar(diff_x, diff_y)
        #convert back to uint8 (0,255) and absolute values
        gradient_mag = cv2.convertScaleAbs(gradient_mag)

        return gradient_mag, theta
    
    if method == 'sobel':
        sobel_x = cv2.Sobel(grey_img, cv2.CV_32F, 1, 0, 3)
        sobel_y = cv2.Sobel(grey_img, cv2.CV_32F, 0, 1, 3)

        gradient_mag, theta = cv2.cartToPolar(sobel_x, sobel_y)
        
        gradient_mag = cv2.convertScaleAbs(gradient_mag)
        
        return gradient_mag, theta

    if method == 'laplacian':
        gradient_mag = cv2.Laplacian(grey_img, cv2.CV_32F, 3)
        
        gradient_mag = cv2.convertScaleAbs(gradient_mag)
        
        return gradient_mag

def visualize_theta(mag, theta):
    #create new image, assign pixel to corresponding color*normalized mag
    row, col = mag.shape
    out_img = np.zeros([row, col, 3], dtype=np.uint8)
    
    for y in range(row):
        for x in range(col):
            
            if theta[x, y] <= (3.14/2):
                color = [255, 0 ,0]
            elif theta[x, y] <= (3.14):
                color = [0, 255 ,0]
            elif theta[x, y] <= (3.14*1.5):
                color = [0, 0 ,255]
            else:
                color = [255, 0, 255]
            pixel = np.multiply(color, float(mag[x, y])/255)
            out_img[x, y] = pixel

    return out_img
    

diff_mag, diff_theta = edge_detection(img, 'diff')
sobel_mag, sobel_theta = edge_detection(img, 'sobel')
laplacian_mag = edge_detection(img, 'laplacian')

cv2.imshow('diff_mag', diff_mag)
cv2.imshow('sobel_mag', sobel_mag)
cv2.imshow('laplacian_mag', laplacian_mag)

cv2.imshow('diff_theta', visualize_theta(diff_mag, diff_theta))
cv2.imshow('sobel_theta', visualize_theta(sobel_mag, sobel_theta))
cv2.waitKey()
