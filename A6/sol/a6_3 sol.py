import os, argparse, sys, math, time
import cv2
import numpy as np
from datetime import datetime
from a6_utils import getTrackingObject, writeCorners, readTrackingData, \
    getBestFitRectangle, writeCorners, drawRegion, rearrangeCorners
from a6_1_sol import draw_cube, project, cube, calibrate

from tracker_andy import Tracker


def main():
    sequences = ['bookI', 'bookII', 'bookIII', 'cereal']
    
    tracker = Tracker()

    calib_src_dir = 'cv_chessboard'  
    calib_img_template = 'image_%d.bmp'
    n_calib_frames = 10

    scale_factor = 1

    # 3D for pattern/book tracked 
    objp = np.zeros((4, 3), np.float32)
    objp = np.float32(
        [[0,0,0],
         [6,0,0],
         [6,6,0],
         [0,6,0]]).reshape(4,3)
   
    # 2D for the 3D points 
    obj_2d_pts = objp[:,:2].reshape(-1, 1, 2)
   
    # 3D object that will be introduced in the scene
    # arrage the object on the pattern the way you like 
    edge_length = 6
    obj_points = cube(edge_length)
        
    # sequence    
    seq_id = 0
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
    K = np.array([[500, 0, 400], [0, 500, 300], [0, 0, 1]]).astype(np.float32)
    #K, dist, images = calibrate(n_calib_frames, calib_src_dir, calib_img_template, vis_delay=1)    

    # process first frame 
    ret, init_img = cap.read()
    if not ret:
        raise IOError("Initial frame could not be read")

    # initialize tracker with the first frame and the initial corners
    tracker.initialize(init_img, init_corners)
    init_pts = np.float32(
        [init_corners[:, 0].tolist(),
         init_corners[:, 1].tolist(),
         init_corners[:, 2].tolist(),
         init_corners[:, 3].tolist()]
    ).reshape(-1, 1, 2)
    
    # this part is similar to ex1
    # solve for the R rotation and t translation using the 3D poits from the book and the tracked points 
    _ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, init_pts, K,np.array([]))
    # project object (cube) points in the image using K, R, t
    obj_img_points = project(obj_points, K, np.array([]), rvecs, tvecs)
    # draw object using the projected points  
    init_img_cube = draw_cube(init_img, obj_img_points)
        
    frame_id = 1

    # draw tracking 
    drawRegion(init_img, init_corners, result_color, thickness)
    cv2.imshow('superimposed_img', init_img_cube)
    #cv2.waitKey(0)
    tracking_fps = []
    mean_fps = 0

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
        curr_pts = np.float32(
            [tracker_corners[:, 0].tolist(),
             tracker_corners[:, 1].tolist(),
             tracker_corners[:, 2].tolist(),
             tracker_corners[:, 3].tolist()]
        ).reshape(-1, 1, 2)

        # this part is similar to ex1
        # solve for the R rotation and t translation using the 3D poits from the book and the tracked points         
        _ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, curr_pts, K,np.array([]))
        # project object (cube) points in the image using K, R, t
        obj_img_points = project(obj_points, K, np.array([]), rvecs, tvecs)
        # draw object using the projected points 
        curr_img_cube = draw_cube(curr_img, obj_img_points)

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
            cv2.imshow('superimposed_img', curr_img_cube)
            
            if cv2.waitKey(1) == 27:
                break
                # print 'curr_error: ', curr_error

        frame_id += 1

        if no_of_frames > 0 and frame_id >= no_of_frames:
            break

    print('mean_fps: {}'.format(mean_fps))


main()
