import os, argparse, sys, math, time
import cv2
import numpy as np
from datetime import datetime
from a6_utils import getTrackingObject, writeCorners, readTrackingData, \
    getBestFitRectangle, writeCorners, drawRegion, rearrangeCorners
from a6_1_ab import draw_cube, project, cube, calibrate

from tracker_cv import Tracker
#from tracker_mtf import Tracker

def main():
    sequences = ['bookI', 'bookII', 'bookIII', 'cereal']
    
    tracker = Tracker()

    calib_src_dir = 'cv_chessboard'  
    calib_img_template = 'image_%d.bmp'
    n_calib_frames = 10

    scale_factor = 1

    # 3D for pattern/book tracked 
    # objp = ...
   
   
    # 3D object that will be introduced in the scene
    # arrage the object on the pattern the way you like 
    # ex cube 
    # obj_points = ...
        
    # sequence    
    seq_id = 2
    img_template = 'img%03d.jpg'
    show_tracking_output = 1
    
    no_of_frames = 0
    
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
    thickness = 2
    result_color = (0, 0, 255)

    print('no_of_frames: {}'.format(no_of_frames))
    
    # extract the true corners in the first frame and place them into a 2x4 array
    init_corners = [ground_truth[0, 0:2].tolist(),
                    ground_truth[0, 2:4].tolist(),
                    ground_truth[0, 4:6].tolist(),
                    ground_truth[0, 6:8].tolist()]
    init_corners = np.array(init_corners).T

    init_corners = rearrangeCorners(init_corners)

    # get calibration (you can just copy result from ex1)
    # dist culd be null array
    # the method is not very sensitive to calibration 
    # K = ...
    # dist = ...
    
    # process first frame 
    ret, init_img = cap.read()
    if not ret:
        raise IOError("Initial frame could not be read")

    # initialize tracker with the first frame and the initial corners
    tracker.initialize(init_img, init_corners)
    # image points (from init_corners) in the format required by  cv2.getPerspectiveTransform
    # make sure you work with floats
    # init_pts = ...
    
    # this part is similar to ex1
    # solve for the R rotation and t translation 
    # using the 3D poits from the book and the tracked points and calibration K,dist
    # ...
    # project object (cube) points in the image using K, R, t
    # obj_img_points = ...
    # draw object using the projected points - init_points 
    # init_img_cube = ...
    cv2.imshow('AR img', init_img_cube) 
    
    # draw tracking 
    drawRegion(init_img, init_corners, result_color, thickness)
    tracking_fps = []
    mean_fps = 0
    frame_id = 1
    
    # loop through the other frames
    while True:
        ret, curr_img = cap.read()
        if not ret:
            print("Frame {} could not be read".format(frame_id))
            break

        start_time = time.clock()
        # update the tracker with the current frame
        tracker_corners = tracker.update(curr_img)
        end_time = time.clock()

        # get position of tracker for computing pose 
        # curr_pts = ...

        # this part is similar to ex1
        # solve for the R rotation and t translation using the 3D poits from the book and the tracked points         
        # ...
        # project object (cube) points in the image using K, R, t
        # obj_img_points = ...
        # draw object using the projected points 
        # curr_img_cube = ...

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
            # display the image
            cv2.imshow(' AR img', curr_img_cube)
            
            if cv2.waitKey(1) == 27:
                break
                # print 'curr_error: ', curr_error

        frame_id += 1

        if no_of_frames > 0 and frame_id >= no_of_frames:
            break

    print('mean_fps: {}'.format(mean_fps))


main()
