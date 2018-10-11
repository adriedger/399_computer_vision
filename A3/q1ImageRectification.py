#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2 

img = cv2.imread('data_q1_rect/book1.jpg', 0)

def get_corners(img):
    
    print "Double-click on the four corners of the quadrilateral"
    pts = []
    # mouse callback function, runs this function on mouse double-click
    def mouse_dblclk(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK: 
            pts.append((x, y))
            cv2.circle(img, (x,y), 25, 0, -1)
    
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', mouse_dblclk)
    # runs until there are 4 double-clicks   
    while 1:
        cv2.imshow('input', img)
        if cv2.waitKey(20) and len(pts) == 4:
            break
    
    return np.asarray(pts)

def get_affine(pts1, pts2):

    # shape 2x4 pts1 into 6x2n
    # reminder: dimensions are y,x for numpy
    row, col = pts1.shape
    M = np.zeros([2*row, 6], dtype=np.float32)
    #print M
    for y in range(row):
        M[2*y, 0:2] = pts1[y, :]
        M[2*y, 2] = 1
        M[2*y+1, 3:5] = pts1[y, :]
        M[2*y+1, 5] = 1
    
    #print M
    # shape 2x4 pts2 into 1x2n
    N = pts2.reshape((8, 1))
    #print N

    # use least-squares (since M isn't square) to calculate determinants
    x = np.linalg.lstsq(M, N)
    #print x[0]
    return np.reshape(x[0], (2, 3))

#pts = get_corners(img)
a = np.arange(1, 9).reshape((4, 2))
b = np.arange(-9, -1).reshape((4, 2))
print get_affine(a, b)
#print cv2.getAffineTransform(a, b)
