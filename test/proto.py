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

    bf = cv2.BFMatcher(cv2.NORM_L2, False)
    if ftype == 'orb':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, False)

    matches = bf.knnMatch(des1, des2, k=2)
    good_match = []
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good_match.append(m)
    
    matches = good_match

    if len(matches) > 4:
        # Extract location of matches in both images
        pts1 = np.zeros((len(matches), 2), dtype=np.float32)
        pts2 = np.zeros((len(matches), 2), dtype=np.float32)        
        for y, m in enumerate(matches):
            pts1[y, :] = kp1[m.queryIdx].pt
            pts2[y, :] = kp2[m.trainIdx].pt        

        # 0 - reg method using all points RANSAC LMEDS - Least-Median, ransacReprojThreshold
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

        return matches, H, mask

    print "Error: Less than 5 matches were found"
    return 0, 0 


def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homography H'''
    #Calculate resultant image size, then do a translation
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
    return result


def stitch(imgs):

    if len(imgs) < 2:
        print "Need more images"
        return None

    fd = 'orb'
    
    for i in range(len(imgs) - 1):
        
        kp1, des1 = detectAndDescribe(imgs[i], fd)
        print "Keypoints found in train img:", len(kp1)
        kp2, des2 = detectAndDescribe(imgs[i+1], fd)   
        print "Keypoints found in query img:", len(kp2)
        
        matches, H, mask = matchKeypoints(kp2, kp1, des2, des1, 1, fd, 0.75)
        print "Matches found:", len(matches)
        print H

        out = warpTwoImages(imgs[i], imgs[i+1], H)       
        return out

imgs = []
for i in range(1, len(sys.argv)):
    img = cv2.imread(str(sys.argv[i])) 
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(str(sys.argv[i]), img)
    imgs.append(img)

pano = stitch(imgs)
cv2.imshow('pano', pano)
cv2.waitKey()
