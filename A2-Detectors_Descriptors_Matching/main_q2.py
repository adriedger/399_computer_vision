import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr

# 2.1. Harris corner detector 

filename = 'data_q2/checkboard.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
# here is the cfunction call 
# k= 0.04, blockSize = 2
dst = my_harris(img,k,blockSize) 
# should behave similar to 
# dst = cv2.cornerHarris(gray,2,3,0.04)

# To get corner locations : threshold for an optimal value, it may vary depending on the image.
# For visualization - put red color for corners
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#-------------------
# 2.2 SIFT 
img = cv2.imread('data_q2/episcopal_gaudi1.jpg',0)  
img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

# Initiate SIFT detector
# ...
# compute the sift detectors (kp) and descriptors (des) using cv2 functions
# ...

# visualize results
img_d = cv2.drawKeypoints(img,kp,outImage=np.array([]),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFTfeatures',img_d)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('sift_keypoints.jpg',img)


#-------------------
# 2.3 Matching SIFT features 

img1 = cv2.imread('data_q2/episcopal_gaudi1.jpg',0)          # queryImage
img2 = cv2.imread('data_q2/episcopal_gaudi2.jpg',0) # trainImage
img1 = cv2.resize(img1,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
img2 = cv2.resize(img2,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

# Initiate SIFT detector
# ...

# find the keypoints and descriptors with SIFT for the two images
# descriptors returned as des1, des2
# ...

# initialize BFMatcher 
# ...

# 2.3.1 brute force matching : compute and sort matches points into list matches
# ...
# Draw first 10 matches.
img_d = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)
cv2.imshow('SIFTmatches  features-brute force',img_d)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2.3.2 ratio distance ; use knnMatch with k=2 ; 
# gives list of best two matches for each feature
# ...
# Apply ratio test and save good matches in a list of DMatch elements
# ...
# Draw good matches 
img_d = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_match,None,flags=2)
cv2.imshow('SIFTmatches  features',img_d)
cv2.waitKey(0)
cv2.destroyAllWindows()
