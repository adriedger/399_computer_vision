import os, argparse, sys, math, time
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from sol_a7_2 import get3Dcube, project, getTransformMatrices

def getFeaturePoints(img, n_pts, pt_size=5, col=(0, 255, 0), title=None):
    if title is None:
        title = 'Select {} feature points'.format(n_pts)
    cv2.namedWindow(title)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pts = []

    def drawPoints(img, _pts):
        if len(_pts) == 0:
            return
        for i in range(len(_pts)):
            pt = _pts[i]
            cv2.circle(img, pt, pt_size, col, thickness=-1)

    def mouseHandler(event, x, y, flags=None, param=None):
        if len(pts) >= n_pts:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            # temp_img = annotated_img.copy()
            # drawLines(temp_img, pts, title)
        elif event == cv2.EVENT_LBUTTONUP:
            pass
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(pts) > 0:
                print('Removing last point')
                del (pts[-1])
        elif event == cv2.EVENT_RBUTTONUP:
            pass
        elif event == cv2.EVENT_MBUTTONDOWN:
            pass
        elif event == cv2.EVENT_MOUSEMOVE:
            pass

    cv2.setMouseCallback(title, mouseHandler)

    while len(pts) < n_pts:
        key = cv2.waitKey(1)
        if key == 27:
            return None
        # _img = np.copy(img)
        drawPoints(img, pts)
        cv2.imshow(title, img)

    cv2.waitKey(250)
    cv2.destroyWindow(title)
    corners = np.array(pts).T
    return corners


def stereo(f, B, x1, x2, y1, cx, cy):
    Z = np.reshape(f * B / (x1 - x2), (-1, 1))

    X1 = (x1 - cx).reshape((-1, 1))
    X2 = (Z / f).reshape((-1, 1))


    X = np.reshape(np.multiply(X1, X2), (-1, 1))
    Y = np.reshape(np.multiply((-y1 - cy).reshape((-1, 1)), (Z / f).reshape((-1, 1))), (-1, 1))

    P = np.concatenate((X, Y, Z), axis=1)
    return Z, P


def main():
    
    n_pts = 1000    #number of points in the cube point cloud
    plot_pts =1     #flag to toggle plotting of the cube points
    rot_x = 90      #rotation angle in degrees about the X axis
    rot_y = -90      #rotation angle in degrees about the Y axis
    rot_z = 0      #rotation angle in degrees about the Z axis
    f = 300         # focal length - change to see how projection changes 
    trans_x = 2  # Translation along the X axis
    trans_y = 0  # Translation along the Y axis
    trans_z = 0  # Translation along the Z axis
    image_size = [640,480] #Image size as a string, e.g. 640x480
    scale_factor = 1       # scale factor for the cube
    
    n_feature_pts = 8 # number of feature points to be selected by the user
                       # < 0 to load previously saved points from file
    baseline = 0.3     # camera translation between the stereo images

    # generate the cube  
    box = get3Dcube(n_pts)

    # conversion from deg to radians 
    rot_x = np.deg2rad(rot_x)
    rot_y = np.deg2rad(rot_y)
    rot_z = np.deg2rad(rot_z)

    # camera center
    cx, cy = image_size[0] / 2, image_size[1] / 2

    # cemara matrix 
    K = np.array(
        [[f, 0, 0, cx],
         [0, f, 0, cy],
         [0, 0, 1, 0]]
    )

    # first image 
    R1, t1 = getTransformMatrices([trans_x, trans_y, trans_z], [rot_x, rot_y, rot_z])
    proj_box1, I1 = project(box, K, R1, t1, image_size)
    I1 = cv2.cvtColor(I1, cv2.COLOR_GRAY2RGB)

    # second image ; translation along baseline 
    R2, t2 = getTransformMatrices([trans_x, trans_y - baseline, trans_z], [rot_x, rot_y, rot_z])
    proj_box2, I2 = project(box, K, R2, t2, image_size)
    I2 = cv2.cvtColor(I2, cv2.COLOR_GRAY2RGB)


    if n_feature_pts > 0:
        pts1 = getFeaturePoints(I1, n_feature_pts, title='Select {} feature points'.format(n_feature_pts))
        if pts1 is None:
            print('Feature points for image 1 could not be obtained')
            return

        pts2 = getFeaturePoints(I2, n_feature_pts,
                                title='Select corresponding feature points in the same order as before')
        if pts2 is None:
            print('Feature points for image 2 could not be obtained')
            return
        np.savetxt('feature_pts_{}.txt'.format(n_feature_pts), np.concatenate((pts1, pts2), axis=0), fmt='%d')

    elif n_feature_pts < 0:
        n_feature_pts = - n_feature_pts
        pts = np.loadtxt('feature_pts_{}.txt'.format(n_feature_pts), dtype=np.int32)
        if n_feature_pts != pts.shape[1]:
            raise IOError('Invalid feature pts: {}'.format(pts))
        pts1 = pts[:2, :]
        pts2 = pts[2:, :]

    print('pts1:\n{}'.format(pts1))
    print('pts2:\n{}'.format(pts2))

    x1 = pts1[0, :]
    y1 = pts1[1, :]
    x2 = pts2[0, :]

    Z, P = stereo(f, baseline, x1, x2, y1, cx, cy)

    if plot_pts:
        # attach numbers for image points (feature points) 
        for i in range(n_feature_pts):
            pt = pts1[:, i]
            # cv2.circle(I1, pt, pt_size, col, thickness=-1)
            cv2.putText(I1, '{}'.format(i+1), tuple(pt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

            pt = pts2[:, i]
            # cv2.circle(I1, pt, pt_size, col, thickness=-1)
            cv2.putText(I2, '{}'.format(i+1), tuple(pt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

        # the 2 stereo images     
        plt.figure()
        plt.imshow(I1, cmap='gray')
        plt.title('Image 1')

        plt.figure()
        plt.imshow(I2, cmap='gray')
        plt.title('Image 2')

        # original cube 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(box[:, 0], box[:, 1], box[:, 2], c=(0, 0, 1), marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Original cube')
    
        # recosntructed cobe 
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')

        # draw lines between points in this order 1>2>3>4>5>6>7>8>1 
        for i in range(n_feature_pts-1):
            ax2.plot(
                (P[i, 0], P[i+1, 0]),
                (P[i, 1], P[i+1, 1]),
                (P[i, 2], P[i+1, 2]),
                c=(0, 0, 1), marker='.')
            
        ax2.plot(
            (P[-1, 0], P[0, 0]),
            (P[-1, 1], P[0, 1]),
            (P[-1, 2], P[0, 2]),
            c=(0, 0, 1), marker='.')

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Reconstructed cube')
 
        plt.show()

main()
