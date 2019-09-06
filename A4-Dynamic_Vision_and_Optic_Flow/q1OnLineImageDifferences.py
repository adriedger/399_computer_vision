#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

from __future__ import print_function
import numpy as np
import cv2 
import time, os

def getImages(n_frames, diff=0, src_dir=''):
    
    img_seq = []
    if src_dir:
        cap = cv2.VideoCapture(os.path.join(src_dir, 'image_%d.bmp'))
        print('Capturing images from {}'.format(src_dir))
        
    else:
        cap = cv2.VideoCapture(0)
        print('Capturing images from camera')
    
    for i in range(n_frames):
        retval, img = cap.read()
        if(retval):
            print('Frame read success', i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_seq.append(img)
        else:
            print('Failure reading frame', i)
        
    return img_seq


def getDifference(img_seq, threshold):
    
    img_seq_diff = []
    for i in range(len(img_seq) - 1):
        img1 = np.float32(img_seq[i])
        img2 = np.float32(img_seq[i+1])
        
        img_diff = img1-img2
        
        img_diff[img_diff[:, :]<threshold] = 0

        img_seq_diff.append(img_diff)
        
    return img_seq_diff

#img_seq = getImages(32)
img_seq = getImages(32, 0, 'Arm32im')

imgs = getDifference(img_seq, 50)
for c, i in enumerate(imgs):
    cv2.imshow(str(c), i)

cv2.waitKey()
