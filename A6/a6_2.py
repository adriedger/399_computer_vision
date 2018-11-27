import os, argparse, sys, math, time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from a6_utils import getTrackingObject
from a6_utils import getTrackingObject, readTrackingData, getBestFitRectangle, \
    writeCorners, drawRegion, rearrangeCorners

#from tracker_mtf import Tracker
from tracker_cv import Tracker

# generates the AR frame by overlaying the warped book img_1 cover to the curent image img_2
def superimpose(img_1, img_2):
    img_1_gs = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img_1_gs, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    roi = cv2.bitwise_and(img_1, img_1, mask=mask)
    im2 = cv2.bitwise_and(img_2, img_2, mask=mask_inv)
    result = cv2.add(im2, roi)
    return result

def main():
    sequences = ['bookI', 'bookII', 'bookIII', 'cereal']

    tracker = Tracker()

    seq_id = 0
    img_template = 'img%03d.jpg'
    show_tracking_output = 1
    use_dlt = 1
    cover_img = 'cover.jpg'
    thickness = 2
    result_color = (0, 0, 255)

    no_of_frames = 0

    # read first fame 
    if seq_id < 0:
        # from camera
        print('Capturing images from camera')
        cap = cv2.VideoCapture(0)
        print('Done')
        if cap is None:
            print('Camera stream could not be opened')
            sys.exit()
        init_corners = getTrackingObject(cap)
        if init_corners is None:
            raise IOError("Tracking object could not be obtained")
    else:
        # images from one of the recorded sequences
        if seq_id >= len(sequences):
            print('Invalid dataset_id: {}'.format(seq_id))
            sys.exit()

        seq_name = sequences[seq_id]
        print('seq_id: {}'.format(seq_id))
        print('seq_name: {}'.format(seq_name))

        src_fname = os.path.join('../a5_tracking/data/'+seq_name, img_template)
        cap = cv2.VideoCapture(src_fname)
        if cap is None:
            print('Image sequence {} could not be opened'.format(src_fname))
            sys.exit()
        # read the ground truth
        ground_truth_fname = '../a5_tracking/data/ground_truth/'+seq_name + '.txt'
        ground_truth = readTrackingData(ground_truth_fname)
        no_of_frames = ground_truth.shape[0]

        print('no_of_frames: {}'.format(no_of_frames))

        # extract the true corners in the first frame and place them into a 2x4 array
        init_corners = [ground_truth[0, 0:2].tolist(),
                        ground_truth[0, 2:4].tolist(),
                        ground_truth[0, 4:6].tolist(),
                        ground_truth[0, 6:8].tolist()]
        init_corners = np.array(init_corners).T

    # to match the order in which the AR object corners are given 
    init_corners = rearrangeCorners(init_corners)

    # read first frame 
    ret, init_img = cap.read()
    if not ret:
        raise IOError("Initial frame could not be read")
    init_h, init_w = init_img.shape[:2] 
    
    # initialize tracker with the first frame and the initial corners   
    tracker.initialize(init_img, init_corners)
    # image points (from init_corners) in the format required by  cv2.getPerspectiveTransform
    # make sure you work with floats
    # init_pts = ...
    
    # window for displaying the tracking result
    window_name = 'Tracking Result'
    if show_tracking_output:        
        cv2.namedWindow(window_name)

    # read ARobject = cover image and initialize size 
    # cover_img = ...
    # cover_h, cover_w = 
   
    # coordinates of the corners in the AR object 
    # top-l top-r bottom-r bottom-l
    # make sure you work with floats and in the required format for cv2.getPerspectiveTransform
    # cover_pts = ...

    # get homography between cover_pts and init_pts 
    # can use cv2.getPerspectiveTransform or cv2.findHomography
    # cover_hom = ...

    # warp the cover image with the homography cv2.warpPerspective
    # warped_cover = ...

    # tracking result 
    drawRegion(init_img, init_corners, result_color, thickness)
    cv2.imshow(window_name, init_img)

    # AR result 
    superimposed_img = superimpose(warped_cover, init_img)
    cv2.imshow('superimposed_img', superimposed_img)

    # list for accumulating the tracking fps for all the frames
    tracking_fps = []
    frame_id = 1
    mean_fps = 0
    
    # loop thrugh the remaining frames and do the same thing with the current tracked points 
    while True:
        ret, curr_img = cap.read()
        if not ret:
            print("Frame {} could not be read".format(frame_id))
            break

        start_time = time.clock()
        # update the tracker with the current frame
        tracker_corners = tracker.update(curr_img)
        end_time = time.clock()

        # get current points from the tracker result
        # in the required format for cv2.getPerspectiveTransform as floats
        # curr_pts = ...

        # compute homography between \cover_pts to curr_points 
        #cover_hom = ...
    
        # warp cover using homography 
        #warped_cover = ...

        # superimpose warped cover onto the source image
        superimposed_img = superimpose(warped_cover, curr_img)

        # compute the tracking fps
        current_fps = 1.0 / (end_time - start_time)
        mean_fps += (current_fps - mean_fps) / frame_id
        tracking_fps.append(current_fps)

        if show_tracking_output:
            # draw the tracker location
            drawRegion(curr_img, tracker_corners, result_color, thickness)
            # write statistics (error and fps) to the image
            cv2.putText(curr_img, "frame {:d} fps: {:5.2f}({:5.2f})".format(
                frame_id, current_fps, mean_fps), (5, 15),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
            # display tracking result
            cv2.imshow(window_name, curr_img)
            # display AR result 
            cv2.imshow('superimposed_img', superimposed_img)

            if cv2.waitKey(1) == 27:
                break
                # print 'curr_error: ', curr_error

        frame_id += 1

        if no_of_frames > 0 and frame_id >= no_of_frames:
            break

    print('mean_fps: {}'.format(mean_fps))


main()
