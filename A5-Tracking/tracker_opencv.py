#Andre's version

from __future__ import print_function
import sys
import cv2
import numpy as np


class Tracker:
    def __init__(self):
        self.initial_corners = None
        self.current_corners = None
        self.tracker = None

    def getBestFitRectangle(self, corners):
        centroid = np.mean(corners, axis=1)
        mean_half_size = np.mean(np.abs(corners - centroid.reshape((2, 1))), axis=1)

        top_left = np.squeeze(centroid - mean_half_size)
        rect_size = np.squeeze(2 * mean_half_size)

        return top_left[0], top_left[1], rect_size[0], rect_size[1]

    def initialize(self, img, corners):
        # initialize your tracker with the first frame from the sequence and
        # the corresponding corners from the ground truth
        # this function does not return anything
        self.initial_corners = corners
        # ...
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        tracker_type = tracker_types[2]

        if tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            self.tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create() 
        
        #startup tracker returns success code, takes initial frame(img) and bounding box
        #print(corners)
        #x, y, w, h = cv2.minAreaRect(corners)
        x, y, w, h = self.getBestFitRectangle(corners)
        bbox = (x, y, w, h)
        #print(bbox)
        ok = self.tracker.init(img, bbox)
        if not ok:
            print("err in tracker init")

    def update(self, img):
        # update your tracker with the current image and return the current corners
        # ...
        ok, bbox = self.tracker.update(img)
        #print(bbox)
        #x, y, w, h to corner points
        self.current_corners = np.asarray(([bbox[0],bbox[1], bbox[0]+bbox[2],bbox[1], bbox[0]+bbox[2],bbox[1]+bbox[3], bbox[0],bbox[1]+bbox[3]])).reshape(4, 2).T
        
        #print(self.current_corners)
        return self.current_corners
       
