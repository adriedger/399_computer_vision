import sys
import cv2
import numpy as np


class Tracker:
    def __init__(self):
        self.init_corners = None
        self.curr_rect = self.curr_corners = None

        self.init_img = None
        self.template = None
        self.template_h = self.template_w = 0

        # Parameters
        self.gauss_kernel_size = 3
        self.epsilon = 0.001
        self.max_iters = 100
        self.skip = 3  # pixels to skip while sampling
        self.update_template = 0  # whether or not to update the template at each frame (for better efficiency)

        self.show_patches = 0 # might be usful for debugging 
        self._pause = 1

    def getBestFitRectangle(self, corners):
        centroid = np.mean(corners, axis=1)
        mean_half_size = np.mean(np.abs(corners - centroid.reshape((2, 1))), axis=1)

        top_left = np.squeeze(centroid - mean_half_size)
        rect_size = np.squeeze(2 * mean_half_size)

        return top_left[0], top_left[1], rect_size[0], rect_size[1]

    # initialize your tracker with the first frame from the sequence and
    # the corresponding corners from the ground truth
    # this function does not return anything    
    def initialize(self, img, corners):
        self.init_img = img
        self.init_corners = corners
        # convert to grayscale
        if len(self.init_img.shape) == 3:
            self.init_img = cv2.cvtColor(self.init_img, cv2.COLOR_BGR2GRAY)
        # gaussian filter 
        self.init_img = cv2.GaussianBlur(self.init_img, (self.gauss_kernel_size, self.gauss_kernel_size), 0)
        # fit a straingt rectangle to init corners
        x, y, w, h = self.getBestFitRectangle(corners)
        w = int(w) 
        h = int(h)        
        # curr_rect stores curreny location of tracker 
        self.curr_rect = [x, y, w, h]
        # initialize template
        self.template = self.init_img[int(y): int(y + h): self.skip, int(x): int(x + w): self.skip]
        self.template_h, self.template_w = self.template.shape

        if self.show_patches:
            cv2.imshow('template', self.template.astype(np.uint8))

    # update your tracker with the current image and return the current corners
    # this is the function that you need to complete
    def update(self, img):
        # curent frame that you use for update
        self.curr_img = img

        skip = self.skip
        # convert to grayscale 
        if len(self.curr_img.shape) == 3:
            self.curr_img = cv2.cvtColor(self.curr_img, cv2.COLOR_BGR2GRAY)

        # gaussian filter and derivatives 
        # self.curr_img = ...
        # dx = ...
        # dy = ...

        # update tracker until convergence of until max iterations 
        for it in range(self.max_iters):

            # get curent location of tracker from self.curr_rect
            #x = ...
            #y = ...
            #w = ...
            #h = ...

            # blocks of derivatives and image using at subpixel accuracy
            # because the tracker is updated at subpizel accuracy we need to use interpolation
            # use this function cv2.getRectSubPix            
            # size of the tracker (for the interpolatin function 
            rect_center = (x + w / 2.0, y + h / 2.0)
            rect_size = (int(w), int(h))
            # curW = ...
            # dxw = ...
            # dyw = ...

            # downsample for better efficiency
            curW = curW[::skip, ::skip]
            dxw = dxw[::skip, ::skip]
            dyw = dyw[::skip, ::skip]

            # gt difference with template 
            # diff = ...

            # reshape derivative and difference blocks as columns 
            #Idx = ...
            #Idy = ...
            #Idt = ...

            # if some values end up undefined ...
            Idx[np.isnan(Idx)] = 0.0
            Idy[np.isnan(Idy)] = 0.0
            Idt[np.isnan(Idt)] = 0.0

            # make M matrix by concatenating Idx Idy
            # M = ...

            # solve equation system to get u,v
            UV = -np.dot(np.linalg.pinv(M), Idt)
            UV[np.isnan(UV)] = 0.0
            
            # get u,v compomenst from solution 
            u, v = UV.squeeze()

            # this is your update 
            self.curr_rect[0] += u
            self.curr_rect[1] += v

            # draw 
            if self.show_patches:
                cv2.imshow('curW', curW.astype(np.uint8))
                cv2.imshow('dxw', dxw.astype(np.uint8))
                cv2.imshow('dyw', dyw.astype(np.uint8))

                k = cv2.waitKey(1 - self._pause)
                if k == 27:
                    sys.exit(0)
                elif k == 32:
                    self._pause = 1 - self._pause
            
            # calculate if update too small - then we stop 
            l2norm = (u * u) + (v * v)
            if l2norm < self.epsilon:
                break

        # put back the update in the tracker params      
        x, y, w, h = self.curr_rect
        self.curr_corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ]).transpose()
        
        # update template if desired 
        if self.update_template:
            self.init_img = self.curr_img
            self.template = self.init_img[int(y): int(y + h): self.skip, int(x): int(x + w): self.skip]
            self.template_h, self.template_w = self.template.shape
            cv2.imshow('template', self.template.astype(np.uint8))

        return self.curr_corners
