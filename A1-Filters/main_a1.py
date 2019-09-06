import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

image1 = load_image('dog.bmp')
image2 = load_image('cat.bmp')
# make grayscale - keep only green channel
# any other method would work
image1=image1[:,:,2]
image2=image2[:,:,2]
# question 1
# example filter 
s = 7
filter = cv2.getGaussianKernel(ksize=s*4+1,sigma=s)
# make filter 2D matrix 
filter = np.dot(filter, filter.T)

plt.figure(figsize=(4,4)); plt.imshow(filter);

# get the blured image by calling your function 
blurry_dog = my_imfilter(image1, filter)
plt.figure(); plt.imshow((blurry_dog*255).astype(np.uint8));

#--------------------------------------------
# question 2
# set filter size 
highFreqsz = 35;
lowFreqsz = 99;
blur, sharp , hybrid = create_hybrid_image(image1, image2, lowFreqsz, highFreqsz )
# show low high pass images and their FT
plt.subplot(121),plt.imshow(blur, cmap = 'gray')
plt.title(''), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(sharp+0.5, cmap = 'gray')
plt.title(''), plt.xticks([]), plt.yticks([])
plt.show()

# visualize the image pyramid
pG = hybrid.copy()
G = hybrid.copy()
for i in range(4):
    G = cv2.pyrDown(G)
    pad = np.ones((hybrid.shape[0]-G.shape[0],G.shape[1]))
    tmp = np.vstack((pad, G))
    pG = np.hstack((pG,tmp))
cv2.imshow('img', pG.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


