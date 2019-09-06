#Andre Driedger, Jacinda Shulman
#Prof: Dana Cobzas CMPT 399 Computer Vision

from __future__ import print_function
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

    print("Error: Less than 5 matches were found")
    return 0, 0 


def composeImages(imgs, matches):
    
    h, w, _ = imgs[0].shape
    # first image corners
    pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
    ptscloud = pts
    
    # get final canvas dimentions
    for i in range(len(imgs)-1):
        
        pts_ = cv2.perspectiveTransform(pts, matches[i][1])
        ptscloud = np.concatenate((ptscloud, pts_), axis=0)

        [xmin, ymin] = np.int32(ptscloud.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(ptscloud.max(axis=0).ravel() + 0.5)
        print("Canvas Dimentions:", xmin, ymin, xmax, ymax)
        
    # extra translation matrix to move negative pixels(outside of canvas) into canvas
    # same for every image
    t = [-xmin,-ymin]
    Ht = np.array([[1, 0, t[0]],
                   [0, 1, t[1]],
                   [0, 0, 1]])

    # build canvas
    result = np.zeros(((ymax-ymin), (xmax-xmin), 3), dtype=np.uint8)          

    # Loop -
    for i in range(len(imgs)-2, -1, -1):
        print("Warping image:", i)
        # all being warped to same canvas size
        warped = cv2.warpPerspective(imgs[i+1], Ht.dot(matches[i][1]), (xmax-xmin, ymax-ymin))                  
        result = blend_seam(result, warped, t, Ht)

    # apply top image
    first = np.zeros_like(result)
    first[t[1]:h+t[1], t[0]:w+t[0]] = imgs[0]
    result = blend_seam(result, first, t, Ht)  

    return result


# Takes the current pano img (curr_canvas) and the warped img to be applied on top. 
#   Other parameters: translation matrices t and Ht
# -Calculates and blends the seam between them.
# -Returns blended image, ready for the next iteration
def blend_seam(curr_canvas, new_img, t, Ht):
    show_results = 0
    h1, w1, _ = new_img.shape
    h2, w2, _ = curr_canvas.shape
    
    # if the curr_canvas is a blank canvas (first iteration), skip the seam 
    # blending and simply add images together
    if(np.count_nonzero(curr_canvas) == 0):
        curr_canvas[new_img>0] = new_img[new_img>0]        
        return curr_canvas

    # if we're in the middle...
    else:
        # get the seam_alpha img
        seam_alpha = calculate_seam(curr_canvas, new_img)
        
        # image stitching
        result = curr_canvas.copy()
        result[new_img>0] = new_img[new_img>0] 
                
        # get a blurred copy of the image (using median blur to get rid of lines)
        blur = cv2.medianBlur(result,7)

        # copy the blurred pixels over to the result    
        result[seam_alpha>0] = blur[seam_alpha>0]

        if(show_results):
            cv2.imshow('blur', blur)
            cv2.imshow('stitched image after blurring', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result


# takes two images of the same dimensions and calculates the seam
# seam is returned as an image of the same dimensions as src images with 
# seam line pixels set to white
def calculate_seam(curr_canvas, new_img):
    # convert both images to grey blocks
    src1 = curr_canvas.copy()
    src2 = new_img.copy()
    src1[src1>0] = 127
    src2[src2>0] = 127

    # add the images together and multiply by 127
    # this makes overlapping areas white and non-overlapping areas grey
    intersection = src1 + src2
    
    # extract the intersection
    intersection[intersection<150] = 0
    intersection[intersection>150] = 255

    # Calculate seam using edge detection on intersection
    seam_alpha = sobel(intersection)
    #cv2.imshow('seam_alpha', seam_alpha)
    # dilate the seam to be a thicker line and blur it
    k=1
    kernel = np.ones((k,k))
    seam_alpha = cv2.dilate(seam_alpha,kernel,iterations = 1).astype(np.uint8)
    seam_alpha[seam_alpha < 200] = 0                # ensure any stray pixels are set to zero

    #cv2.imshow('seam_alpha2', seam_alpha)
    #cv2.waitKey(0)
    return seam_alpha


def sobel(intersection):
    sobel_x = cv2.Sobel(intersection, cv2.CV_64F, 1, 0, 3, delta=10)
    sobel_y = cv2.Sobel(intersection, cv2.CV_64F, 0, 1, 3, delta=-10)
    sobel_x = np.absolute(sobel_x).astype(np.uint8)
    sobel_y = np.absolute(sobel_y).astype(np.uint8)
    seam_alpha = sobel_x + sobel_y
    return seam_alpha


def stitch(input_imgs):

    if len(input_imgs) < 2:
        print("Need more images")
        return None

    fd = 'orb'
    
    features = []
    for i in range(len(input_imgs)):
        
        kp, des = detectAndDescribe(input_imgs[i], fd)
        print("Keypoints found in img "+str(i)+":", len(kp))
        features.append((kp, des))
    
    pairwise_matches = []
    for i in range(len(features) - 1):
        
        kp1, des1 = features[i] 
        kp2, des2 = features[i+1]
        matches, H, mask = matchKeypoints(kp2, kp1, des2, des1, 1, fd, 0.75)
        print("Matches found on pairwise "+str(i)+"-"+str(i+1)+":", len(matches))
        print(H)
        if i == 0:
            pairwise_matches.append((matches, H))
        else:
            pairwise_matches.append((matches, H.dot(pairwise_matches[i-1][1])))
            print("Relative Homography:")
            print(pairwise_matches[i][1])

    out = composeImages(input_imgs, pairwise_matches)
    return out

print("Sequential Image Stitcher\n")
if len(sys.argv) < 2:
    print("Usage:\npython main.py img1 img2 [...imgN]")
else:
    imgs = []
    for i in range(1, len(sys.argv)):
        img = cv2.imread(str(sys.argv[i])) 
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        imgs.append(img)

    pano = stitch(imgs)
    save_name = 'pano_result'
    cv2.imshow(str(sys.argv[1])+str(sys.argv[-1]), pano)
    cv2.imwrite('./{}.jpg'.format(save_name), pano)
    cv2.waitKey()
