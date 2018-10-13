#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import numpy as np
import cv2 

def detectAndDescribe(img, ftype='sift'):
    
    # intialize SIFT or ORB object, return keypoints and their descriptors
    if ftype == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
        return sift.detectAndCompute(img, None)
    if ftype == 'orb':
        orb = cv2.ORB()
        return orb.detectAndCompute(img, None)


def matchKeypoints(kp1, kp2, des1, des2, mtype, ftype='sift', ratio=0.75):
    
    # Match keypoints of both images
    # brute force matching
    if mtype == 0:
        bf = cv2.BFMatcher(cv2.NORM_L2, True)
        if ftype == 'orb':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)

        matches = matches[:int(len(matches)*ratio)]
        #print 'yo', matches
    
    # ratio distance matching
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, False)
        if ftype == 'orb':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, False)
    
        matches = bf.knnMatch(des1, des2, k=2)
        good_match = []
        for m,n in matches:
            if m.distance < ratio*n.distance:
                good_match.append([m])
        
        matches = good_match

    # Reminder: Result of bf.match(des1,des2) is a list of DMatch objects.
    #   DMatch.distance - Distance between descriptors. The lower, the better it is.
    #   DMatch.trainIdx - Index of the descriptor (which corresponds to the keypoint) in train descriptors
    #   DMatch.queryIdx - Index of the descriptor in query descriptors
    #   DMatch.imgIdx - Index of the train image.
    
    # Homography: Find a perspective transformation between two planes
    # (needs more than 4 matches to work)
    if len(matches) > 4:
        # Extract location of matches in both images
        pts1 = np.zeros((len(matches), 2), dtype=np.float32)
        pts2 = np.zeros((len(matches), 2), dtype=np.float32)        
        for y, m in enumerate(matches):
            #print y, m, m.queryIdx, m.trainIdx
            pts1[y, :] = kp1[m.queryIdx].pt
            pts2[y, :] = kp2[m.trainIdx].pt        

        H, _ = cv2.findHomography(pts1, pts2, 0)

        return matches, H

    print "Error: Less than 5 matches were found"
    return 0, 0 


img1 = cv2.imread('data_q2_panor/seoul1.jpg', 0)
img2 = cv2.imread('data_q2_panor/seoul2.jpg', 0)
cv2.imshow('input1', img1)
cv2.imshow('input2', img2)
kp1, des1 = detectAndDescribe(img1, 'sift')
kp2, des2 = detectAndDescribe(img2, 'sift')
matches, H = matchKeypoints(kp1, kp2, des1, des2, 0, 'sift', 0.75)
print H, H.dtype, H.shape
cv2.waitKey()
