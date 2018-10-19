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
        orb = cv2.ORB_create()
        return orb.detectAndCompute(img, None)


def matchKeypoints(kp1, kp2, des1, des2, mtype, ftype='sift', ratio=0.75):
    
    # Find matches of features of both images
    # brute force matching
    if mtype == 0:
        bf = cv2.BFMatcher(cv2.NORM_L2, True)
        if ftype == 'orb':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)

        matches = matches[:int(len(matches)*ratio)]
    
    # ratio distance matching
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, False)
        if ftype == 'orb':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, False)
    
        matches = bf.knnMatch(des1, des2, k=2)
        good_match = []
        for m, n in matches:
            if m.distance < ratio*n.distance:
                good_match.append(m)
        
        matches = good_match

    # Reminder: Result of bf.match(des1,des2) is a list of DMatch objects.
    #   DMatch.distance - Distance between descriptors. The lower, the better it is.
    #   DMatch.trainIdx - Index of the descriptor (corresponds to the keypoint) in train descriptors
    #   DMatch.queryIdx - Index of the descriptor in query descriptors
    #   DMatch.imgIdx - Index of the train image.
    
    # Find a perspective transformation between two feature sets (planes)
    # (needs more than 4 matches to work)
    print "Matches found:", len(matches)
    if len(matches) > 4:
        # Extract location of matches in both images
        pts1 = np.zeros((len(matches), 2), dtype=np.float32)
        pts2 = np.zeros((len(matches), 2), dtype=np.float32)        
        for y, m in enumerate(matches):
            pts1[y, :] = kp1[m.queryIdx].pt
            pts2[y, :] = kp2[m.trainIdx].pt        

        H, _ = cv2.findHomography(pts1, pts2, cv2.LMEDS)
        print H
        # 0 - reg method using all points, RANSAC, LMEDS - Least-Median       
        return matches, H

    print "Error: Less than 5 matches were found"
    return 0, 0 


def stitch(img1, img2):
    
    # Get keypoints and descriptors in each image 
    kp1, des1 = detectAndDescribe(img1, 'sift')
    print "Keypoints found in first img:", len(kp1)
    kp2, des2 = detectAndDescribe(img2, 'sift')   
    print "Keypoints found in second img:", len(kp2)
    # Get matches, homography from img2 -> img1     
    matches, H = matchKeypoints(kp2, kp1, des2, des1, 1, 'sift', 0.75)    
    # Visualize matches
    imgMatches = cv2.drawMatches(img2, kp2, img1, kp1, matches, None)
    
    #estimate dimentions of combined image
    out_x = img1.shape[1] + img2.shape[1]
    out_y = img1.shape[0] + img2.shape[0]
    # Do perspective tranformation on second image
    img2warp = cv2.warpPerspective(img2, H, (out_x, out_y))
    cv2.imshow('warp', img2warp)
    # Init master image
    masterImg = np.zeros((out_y, out_x, 3), np.uint8)
    # Apply first image from left->right
    masterImg[:img1.shape[0], :img1.shape[1]] = img1
    # Create mask to dull pixels that will be combined
    mask = np.ones(img2warp.shape, np.float32)
    mask[(img2warp>0)*(masterImg>0)] = 2.0 #**here**
    # Add images, apply mask
    masterImg = (img2warp.astype(np.float32) + masterImg.astype(np.float32))/mask
    #REMINDER: Homography calculates translation of img2 relative to img1
    
    return masterImg.astype(np.uint8), imgMatches

#img1 = cv2.imread('data_q2_panor/macew1.jpg', 0)
#img2 = cv2.imread('data_q2_panor/macew6.jpg', 0)
img1 = cv2.imread('../DJI_0003.JPG')
img2 = cv2.imread('../DJI_0004.JPG')
img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
cv2.imshow('input1', img1)
cv2.imshow('input2', img2)
pano, matches = stitch(img1, img2)
cv2.imshow('matches', matches)
cv2.imshow('pano', pano)
cv2.waitKey()
