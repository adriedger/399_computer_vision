import sys
import cv2
import numpy as np
class Tracker:

    def __init__(self):
        self.currentCorners = None
        self.firstFrame = np.zeros((2,4)).astype(np.float32)
        self.previousFrame = None
        self.previousImg = None

    def initialize(self, img, corners):
        # initialize your tracker with the first frame from the sequence and
        # the corresponding corners from the ground truth
        # this function does not return anything

        # Initialize tracker for the first frame   
        # Upper left X
        self.firstFrame[0][0] = corners[0][0] 
        # Upper left Y
        self.firstFrame[0][1] = corners[1][0]
        # Upper right X
        self.firstFrame[0][2] = corners[0][1]
        # Upper right Y
        self.firstFrame[0][3] = corners[1][1]
        # Lower right X
        self.firstFrame[1][0] = corners[0][2]
        # Lower right Y
        self.firstFrame[1][1] = corners[1][2]
        # Lower left X
        self.firstFrame[1][2] = corners[0][3]
        # Lower left Y
        self.firstFrame[1][3] = corners[1][3] 		
    
        self.previousImg = img.copy()
        self.previousFrame = cv2.cvtColor(self.previousImg, cv2.COLOR_BGR2GRAY)        

    def update(self, img):
        # update your tracker with the current image and return the current corners    
        self.firstFrame = np.reshape(self.firstFrame, (4,1,2))
        lk_params = dict( winSize  = (29,29), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        nextFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.previousFrame, nextFrame, self.firstFrame, None, **lk_params)
        self.current_corners = p1[st==1].T
        self.previousFrame = nextFrame.copy()
        
        reshaped_p1 = np.reshape(p1, (2,4)) 
        self.firstFrame = reshaped_p1
        
        return self.current_corners