import cv2
import time, argparse, os, sys
import numpy as np


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

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    corners_3d = np.zeros((6 * 7, 3), np.float32)
    corners_3d[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

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
        ret, corners = cv2.findChessboardCorners(img_gs, (7, 6), None)
        # If found, add object points, image points (after refining them)
        if ret:      
            corners2 = cv2.cornerSubPix(img_gs, corners, (11, 11), (-1, -1), criteria)

            obj_points.append(corners_3d)
            img_points.append(corners)
            images.append(img_gs)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow('detected corners', img)
            cv2.waitKey(vis_delay)

            img_id += 1

        else:
            print('Corner detection failed in image {}'.format(img_id))
                
    if not images:
        raise IOError('No valid images found for calibration')

    img = images[0]
    ret2, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, images[0].shape[::-1], None, None)
    
    h, w = img.shape[:2]
    refined_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    for img_id, gray_img in enumerate(images):
        # undistort
        dst = cv2.undistort(gray_img, K, dist, None, refined_K)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        # cv2.imwrite('calib_result_{}.png'.format(img_id), dst)

        cv2.imshow('undistorted img', dst)
        cv2.waitKey(vis_delay)

    cv2.destroyWindow('undistorted img')
    cv2.destroyWindow('detected corners')

    return refined_K, dist, images


def cube(edge_length):
    c = edge_length / 2.0
    return np.float32([
        [0, 0, 0],
        [0, c, 0],
        [c, c, 0],
        [c, 0, 0],
        [0, 0, -c],
        [0, c, -c],
        [c, c, -c],
        [c, 0, -c]
    ])


def project(points, K, dist, R, t):
    proj_pts, jac = cv2.projectPoints(points, R, t, K, dist)

    return proj_pts


def draw_cube(img, cube_img_points):
    imgpts = np.int32(cube_img_points).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    # cv2.imshow('cube img', img)
    # cv2.waitKey(0)

    return img


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

            # Find the rotation and translation vectors.
            _ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, K, dist)

            # print('res; {}'.format(res))

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
