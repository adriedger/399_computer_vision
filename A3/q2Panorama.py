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
    
    # Find matches of features of both images
    # brute force matching
    if mtype == 0:
        bf = cv2.BFMatcher(cv2.NORM_L2, True)
        if ftype == 'orb':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)

        matches = matches[:int(len(matches)*ratio)]
        #matches = matches[:300]
    
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
    #print 'yo', matches, len(matches)
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


def stitch(img1, img2):
    
    # Get keypoints and descriptors in each image 
    kp1, des1 = detectAndDescribe(img1, 'sift')
    kp2, des2 = detectAndDescribe(img2, 'sift')   
    # Get matches     
    matches, H = matchKeypoints(kp1, kp2, des1, des2, 1, 'sift', 0.75)    
    
    # Visualize matches
    imgMatches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
    cv2.imshow('matches', imgMatches)
    
    # Dimensions of combined image adds widths but keeps max height
    out_x = img1.shape[1] + img2.shape[1]
    out_y = max(img1.shape[0], img2.shape[0])
    # Do perspective tranformation on first image (does not work for some)
    out1 = cv2.warpPerspective(img1, H, (out_x, out_y))
    cv2.imshow('warp', out1)
    # init basis for composed image
    out2 = np.zeros((out_y, out_x), np.uint8)
    # apply second image on rightmost area
    out2[:img2.shape[0], :img2.shape[1]] = img2
    # create mask to dull pixels that will be combined
    mask = np.ones(out1.shape, np.float32)
    mask[(out1>0)*(out2>0)] = 2.0 #**here**
    # add images, apply mask
    out2 = (out1.astype(np.float32) + out2.astype(np.float32))/mask
    cv2.imshow('out2', out2.astype(np.uint8))
    
    return out2, imgMatches


img1 = cv2.imread('data_q2_panor/macew1.jpg', 0)
img2 = cv2.imread('data_q2_panor/macew2.jpg', 0)
cv2.imshow('input1', img1)
cv2.imshow('input2', img2)
#kp1, des1 = detectAndDescribe(img1, 'sift')
#kp2, des2 = detectAndDescribe(img2, 'sift')
#matches, H = matchKeypoints(kp1, kp2, des1, des2, 0, 'sift', 0.75)
#matches, H = matchKeypoints(kp1, kp2, des1, des2, 1, 'sift', 0.75)
#print H, H.dtype, H.shape
pano, matches = stitch(img1, img2)
#cv2.imshow('matches', matches)
#cv2.imshow('pano', pano)
cv2.waitKey()
