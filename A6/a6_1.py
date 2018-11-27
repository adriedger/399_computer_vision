import cv2
import time, argparse, os, sys
import numpy as np

# calibrates the camera and returns the undistorsed image 
# results are more accurate if several images of the pattern are used 
# if only one image is used n_frame = 1
# src_dir - source directory with the caibration images
# if empty images are grabbed from camera 
def calibrate(n_frames, src_dir='', img_template='image_%d.jpg', vis_delay=500):
    print('Calibrating on {} images'.format(n_frames))

    if src_dir:
        print('Capturing images from {}'.format(src_dir))
        cap = cv2.VideoCapture(os.path.join(src_dir, img_template))
    else:
        print('Capturing images from camera')
        cap = cv2.VideoCapture(0)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare 3D object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # depends on how many squares your pattern has 
    # I included the opencv calibration sequence but you should take your own also
    # in the opencv sequence pattren is 6x7
    
    # corners_3d = ...
    

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    images = []

    img_id = 0
    while img_id < n_frames:

        ret, img = cap.read()
        if not ret:
            print('Capture of Image {} was unsuccessful'.format(img_id + 1))
            break


        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # use function ret, corners = cv2.findChessboardCorners
        # ret, corners = ...
        
        # If found, add object points, image points (after refining them)
        if ret:     
            # refine to subpixel with cv2.cornerSubPix
            # corners2 = ...

            obj_points.append(corners_3d)
            img_points.append(corners)
            images.append(img_gs)

            # Draw and display the corners
            # use drawChessboardCorners
   
            img_id += 1

        else:
            print('Corner detection failed in image {}'.format(img_id))
                
    if not images:
        raise IOError('No valid images found for calibration')

    # calibrate camera from obj_points and img_points
    # ret2, K, dist, rvecs, tvecs = cv2.calibrateCamera(...)
    
    # get optimal params of undistorsed images 
    # refined_K, roi = cv2.getOptimalNewCameraMatrix

    # display undistorsed images
    for img_id, gray_img in enumerate(images):
        # undistort
        dst = cv2.undistort(gray_img, K, dist, None, refined_K)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        cv2.imshow('undistorted img', dst)
        cv2.waitKey(vis_delay)

    cv2.destroyAllWindows() 

    return refined_K, dist, images

# define the corners of a 3D cube 
# should return a 8x3 matrix of 3D coords 
def cube(edge_length):
# ...


# project 3D points using K, R, t and dist params 
# retruns projected points 
def project(points, K, dist, R, t):
# ...
# return proj_points

# draws a cube defined by projected points cube_img_points in img 
# retruns the image with the cube 
def draw_cube(img, cube_img_points):
# ...
#    return img


def main():
    src_dir = 'cv_chessboard'
    n_frames = 10
    img_template = 'image_%d.jpg'
    edge_length = 6 
 
    # calibrate camera from n_frames 
    K, dist, images = calibrate(n_frames, src_dir, img_template)
    points = cube(edge_length)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    _pause = 0
    for img_id, gray_img in enumerate(images):
        ret, corners = cv2.findChessboardCorners(gray_img, (7, 6), None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors in current image 
            # using 3D-2D correspondances objp, corners2 and calibration K, dist
            # use cv2.solvePnPRansac
            #_ret, rvecs, tvecs, inliers = ...

            cube_img_points = project(points, K, dist, rvecs, tvecs)

            img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            img = draw_cube(img, cube_img_points)

            cv2.imshow('cube img', img)
            k = cv2.waitKey(500 * (1 - _pause)) & 0xff

            if k == 27:
                break
            elif k == 32:
                _pause = 1 - _pause

    cv2.destroyAllWindows()

main()
