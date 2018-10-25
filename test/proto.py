#Andre Driedger 1805536
#Prof: Dana Cobzas CMPT 399 Computer Vision

import sys
import numpy as np
import cv2 

def detectAndDescribe(img, ftype='sift'):
    
    # intialize SIFT or ORB object, return keypoints and their descriptors
    if ftype == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
        return sift.detectAndCompute(img, None)
    if ftype == 'orb':
        orb = cv2.ORB_create(10000)
        return orb.detectAndCompute(img, None)
    if ftype == 'surf':
        surf = cv2.xfeatures2d.SURF_create(300)
        return surf.detectAndCompute(img, None)


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
    if len(matches) > 4:
        # Extract location of matches in both images
        pts1 = np.zeros((len(matches), 2), dtype=np.float32)
        pts2 = np.zeros((len(matches), 2), dtype=np.float32)        
        for y, m in enumerate(matches):
            pts1[y, :] = kp1[m.queryIdx].pt
            pts2[y, :] = kp2[m.trainIdx].pt        

        # 0 - reg method using all points, RANSAC, LMEDS - Least-Median       
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

        return matches, H

    print "Error: Less than 5 matches were found"
    return 0, 0 


def stitch(imgs):

    if len(imgs) < 2:
        print "Need more images"
        return None

    fd = 'orb'
    
    #**do composing seperately**
    y, x, _ = imgs[0].shape
    masterImg = cv2.copyMakeBorder(imgs[0], y/4, y/4, x/4, x/4, cv2.BORDER_CONSTANT, 0)
    out_x = masterImg.shape[1]
    out_y = masterImg.shape[0]
    #masterImg = np.zeros((out_y, out_x, 3), np.uint8)
    #masterImg[:start_img.shape[0], :start_img.shape[1]] = start_img
    #**pre-estimate padding before calculating each homography

    for i in range(len(imgs) - 1):
        
        train_img = cv2.copyMakeBorder(imgs[i], y/4, y/4, x/4, x/4, cv2.BORDER_CONSTANT, 0)
        kp1, des1 = detectAndDescribe(train_img, fd)
        print "Keypoints found in train img:", len(kp1)
        kp2, des2 = detectAndDescribe(imgs[i+1], fd)   
        print "Keypoints found in query img:", len(kp2)
        
        matches, H = matchKeypoints(kp2, kp1, des2, des1, 1, fd, 0.75)
        print "Matches found:", len(matches)
        print H
        
        if i > 0:
            H = np.matmul(H, pre_H)
        pre_H = H
        
        # Visualize matches
        #imgMatches = cv2.drawMatches(img2, kp2, img1, kp1, matches, None)         
        
        imgWarp = cv2.warpPerspective(imgs[i+1], H, (out_x, out_y))
        
        mask = np.ones(imgWarp.shape, np.float32)
        mask[(imgWarp>0)*(masterImg>0)] = 2.0 #**here**
        
        masterImg = (imgWarp.astype(np.float32) + masterImg.astype(np.float32))/mask
    
    return masterImg.astype(np.uint8)

imgs = []
for i in range(1, len(sys.argv)):
    img = cv2.imread(str(sys.argv[i])) 
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(str(sys.argv[i]), img)
    imgs.append(img)

pano = stitch(imgs)
cv2.imshow('pano', pano)
cv2.waitKey()
