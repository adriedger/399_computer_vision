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
    #runs until there are 4 double-clicks   
    while 1:
        cv2.imshow('input', img)
        if cv2.waitKey(20) and len(pts) == 4:
            break
    
    return np.asarray(pts)

def get_affine(pts1, pts2):
    #shape 2x4 pts1 into 6x2n

    return 

pts = get_corners(img)
print pts

