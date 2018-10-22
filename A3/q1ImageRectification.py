#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2 

img1 = cv2.imread('data_q1_rect/book1.jpg', 0)
img2 = cv2.imread('data_q1_rect/scanned-form.jpg', 0)

def get_corners(img):
    
    print "Double-click on the four corners of the quadrilateral."
    print "Must be clockwise from top left."
    pts = []
    img_copy = np.copy(img)
    # mouse callback function, runs this function on mouse double-click
    def mouse_dblclk(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK: 
            pts.append((x, y))
            cv2.circle(img_copy, (x,y), 20, 0, -1)
    
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', mouse_dblclk)
    # runs until there are 4 double-clicks   
    while 1:
        cv2.imshow('input', img_copy)
        if cv2.waitKey(20) and len(pts) == 4:
            cv2.imshow('input', img_copy)
            break
    
    # data type needs to be float for cv funtions
    return np.asarray(pts).astype(np.float32)

def get_affine(pts1, pts2):

    # shapes 2x4 pts1 into 6x2n
    # **reminder: dimensions are y,x for numpy**
    row, col = pts1.shape
    M = np.zeros([2*row, 6], dtype=np.float32)
    for y in range(row):
        M[2*y, 0:2] = pts1[y, :]
        M[2*y, 2] = 1
        M[2*y+1, 3:5] = pts1[y, :]
        M[2*y+1, 5] = 1
    
    # shape 2x4 pts2 into 1x2n
    N = pts2.reshape((8, 1))

    # use least-squares (since M isn't square) to calculate determinants
    x = np.linalg.lstsq(M, N)
    
    return np.reshape(x[0], (2, 3))

def rectify(img):
    
    y, x = img.shape
    # **reminder: dimensions are x, y from top corner in cv2**
    pts1 = get_corners(img)
    pts2 = np.array([[10, 10], [10, y-10], [x-10, y-10], [x-10, 10]], np.float32)
    
    # shape is 3x2
    det1 = get_affine(pts1, pts2)
    # shape is 3x3
    det2 = cv2.getPerspectiveTransform(pts1, pts2)
    
    return cv2.warpAffine(img, det1, (x, y)), cv2.warpPerspective(img, det2, (x, y))

out1, out2 = rectify(img1)
cv2.imshow('affine_book', out1)
cv2.imshow('homography_book', out2)
out3, out4 = rectify(img2)
cv2.imshow('affine_form', out3)
cv2.imshow('homography_form', out4)
cv2.waitKey()
