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


# computes point on a grid of block_size for an image of size [img_width,img_height]**Dana's code
def getGridPoints(img_width,img_height,block_size):
    _X = np.asarray(range(0, img_width - 1, block_size)) + block_size // 2
    _Y = np.asarray(range(0, img_height - 1, block_size)) + block_size // 2

    [X, Y] = np.meshgrid(_X, _Y)
    return X,Y


def computeOpticalFlow_opencv(img, img_old, block_size, lk_params):
    
    img_height, img_width = img.shape[:2]
    X, Y = getGridPoints(img_width, img_height, block_size)
    n, m = X.shape
    
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    p0 = np.concatenate((X, Y), axis=1)
    p0 = p0.astype(np.float32)
    p0 = p0.reshape(-1, 1, 2)
    
    # call opencv function cv2.calcOpticalFlowPyrLK
    p1, st, err = cv2.calcOpticalFlowPyrLK(img_old, img, p0, None, **lk_params)
    # compute U, V optic flow vectors from p1 (new position) and p0 (old position) 
    p1[st==0] = 0
    #print(p1[:, 0])
    U = p1[:, :, 0]
    V = p1[:, :, 1]
    #print(U, V)
    p0[st==0] = 0
    X = p0[:, :, 0]
    Y = p0[:, :, 1]
    #print(X, Y)
    # Returns 1d arrays of coords
    return U, V, X, Y


def draw_flow(img, U, V, X, Y):
    
    # groups coords into array of 4 1d matrixes, transposes, then rearanges into array of 2x2 line vectors
    lines = np.asarray([X, Y, U, V]).T.reshape(-1, 2, 2)
    # convert vectors to ints for visualization
    lines = np.int32(lines + 0.5) 
    #print(lines)
    # start color image
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # draw green lines
    vis = cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        # draw grid circles
        vis = cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis



#img_seq = getImages(32, 0, 'Arm32im')
img_seq = getImages(32)

img_seq_diff = getDifference(img_seq, 50)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
block_size = 10


U, V, X, Y = computeOpticalFlow_opencv(img_seq[11], img_seq[10], block_size, lk_params)
test_img = draw_flow(img_seq[11], U, V, X, Y)
#cv2.imshow('test', test_img)

#video = cv2.VideoWriter('test.mp4', -1, 1, img_seq[0].shape[:2])
for i in range(31):
    U, V, X, Y = computeOpticalFlow_opencv(img_seq[i+1], img_seq[i], block_size, lk_params)
    img_flow = draw_flow(img_seq[i+1], U, V, X, Y)
    cv2.imshow('test', img_flow)
    #video.write(img_flow)
    cv2.waitKey(100)

#cv2.waitKey()
#video.release()
