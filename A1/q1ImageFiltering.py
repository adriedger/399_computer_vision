#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2

def my_imfilter(img, f):
    rows, cols = img.shape
    frows, fcols = f.shape
    
    if frows%2 == 0 or fcols%2 == 0:
        print "Error: Filter matrix row or column is even. Cannot blur. Must be odd."
        return

    padded_grey_img = cv2.copyMakeBorder(img, frows/2, frows/2, fcols/2, fcols/2, cv2.BORDER_CONSTANT, 0)    
    out_img = np.zeros([rows, cols, 1], dtype=np.uint8)
    
    for y in range(rows):
        for x in range(cols):
            #snapshots of same dimensions as filter
            weighted = np.multiply(f, padded_grey_img[y:y+frows, x:x+fcols])
            pixel = np.sum(weighted)
            out_img[y][x] = pixel

    return out_img

def test():
    img = cv2.imread('data/lena.tif')
    s = 7

    f = cv2.getGaussianKernel(s*4+1, -1)
    #an array of s*4+1 by 1 coefficients
    #multiply to get dot product i.e. a square matrix
    f = np.dot(f, f.T)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    out_img = my_imfilter(grey_img, f)

    cv2.imshow('grey_lena', grey_img)
    cv2.imshow('blur_lena', out_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('out_data/blur_lena.tif', out_img)

