import os, argparse, sys, math, time
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as plt


def get3Dcube(n_pts):
    x = np.random.rand(n_pts, 1) - 0.5
    allz = np.zeros((n_pts, 1)) - 0.5
    box = np.concatenate(
        (
            np.concatenate((x, allz, allz), axis=1),
            np.concatenate((allz, x, allz), axis=1),
            np.concatenate((x, allz + 1, allz), axis=1),
            np.concatenate((allz + 1, x, allz), axis=1),
            np.concatenate((allz, allz, x), axis=1),
            np.concatenate((allz, allz + 1, x), axis=1),
            np.concatenate((allz + 1, allz + 1, x), axis=1),
            np.concatenate((allz + 1, allz, x), axis=1),
            np.concatenate((x, allz, allz + 1), axis=1),
            np.concatenate((allz, x, allz + 1), axis=1),
            np.concatenate((x, allz + 1, allz + 1), axis=1),
            np.concatenate((allz + 1, x, allz + 1), axis=1),
        ),
        axis=0)
    return box


def applyTransform(T, p1):
    allo = np.ones((p1.shape[0], 1))

    p2 = np.concatenate((p1, allo), axis=1)
    p2 = np.matmul(T, p2.transpose()).transpose()

    p2 = np.concatenate((
        np.reshape(p2[:, 0] / p2[:, 2], (-1, 1)),
        np.reshape(p2[:, 1] / p2[:, 2], (-1, 1))
    ),
        axis=1)

    return p2


def getTransformMatrices(t, r):
    t = np.array(
        [[1, 0, 0, t[0]],
         [0, 1, 0, t[1]],
         [0, 0, 1, t[2]],
         [0, 0, 0, 1]]
    )
    rx = r[0]
    ry = r[1]
    rz = r[2]

    rot_x = np.array(
        [[1, 0, 0, 0],
         [0, np.cos(rx), -np.sin(rx), 0],
         [0, np.sin(rx), np.cos(rx), 0],
         [0, 0, 0, 1]]
    )
    rot_y = np.array(
        [[np.cos(ry), 0, np.sin(ry), 0],
         [0, 1, 0, 0],
         [-np.sin(ry), 0, np.cos(ry), 0],
         [0, 0, 0, 1]]
    )
    rot_z = np.array(
        [[np.cos(rz), -np.sin(rz), 0, 0],
         [np.sin(rz), np.cos(rz), 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    )
    R = np.matmul(np.matmul(rot_z, rot_y), rot_x)

    return R, t


def project(P, K, R, t, image_size=(640, 480)):
    T = np.matmul(R, t)
    pers_final = np.matmul(K, T)

    p = applyTransform(pers_final, P)

    I = np.full((image_size[1], image_size[0]), 255, dtype=np.uint8)
    
    valid_idx = np.logical_and(
        np.logical_and(p[:, 0] < image_size[0], p[:, 0] >= 0),
        np.logical_and(p[:, 1] < image_size[1], p[:, 1] >= 0)
    )
    p_idx = p[valid_idx, :].astype(np.int32)    
    p_idx = p.astype(np.int32)

    I[p_idx[:, 1], p_idx[:, 0]] = 0
    return p, I


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
    
    box = get3Dcube(n_pts)

    rot_x = np.deg2rad(rot_x)
    rot_y = np.deg2rad(rot_y)
    rot_z = np.deg2rad(rot_z)

    #try:
    #    image_size = [int(k) for k in image_size.split('x')]
    #except BaseException as e:
    #    print('Error in parsing image_size {}: {}'.format(image_size, e))
    #    return

    cx, cy = image_size[0] // 2, image_size[1] // 2

    K = np.array(
        [[f, 0, 0, cx],
         [0, f, 0, cy],
         [0, 0, 1, 0]]
    )

    R, t = getTransformMatrices([trans_x, trans_y, trans_z], [rot_x, rot_y, rot_z])

    box *= scale_factor

    box_new, I = project(box, K, R, t, image_size)


    if plot_pts:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(box[:, 0], box[:, 1], box[:, 2], c=(0, 0, 1), marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.figure()
        plt.subplot(1, 1, 1)
        plt.scatter(box_new[:, 0], box_new[:, 1], c=(0, 0, 1), marker='.')
        plt.grid(1)

        plt.figure()
        plt.imshow(I, cmap='gray')

        plt.show()


#main()