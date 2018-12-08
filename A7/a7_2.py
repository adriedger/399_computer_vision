# Jacinda Shulman
# 3050764

import os, argparse, sys, math, time
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as plt

# defines the cube 
def get3Dcube(n_pts):
    # generate n_pts random points in [-0.5,0.5]
    x = np.random.rand(n_pts, 1) - 0.5
    # n_pts of -0.5
    allz = np.zeros((n_pts, 1)) - 0.5
    # make each edge by concatenating x and allz 
    # then concatenate all 12 edges resulting in a n_points*12x3 matrix
    box = np.concatenate(
        (
            # Bottom side - z stays constant
            np.concatenate((x, allz, allz), axis=1),
            np.concatenate((x, allz+1, allz), axis=1), 
            np.concatenate((allz, x, allz), axis=1),
            np.concatenate((allz+1, x, allz), axis=1),
            
            # Top side - z stays constant (allz+1)
            np.concatenate((x, allz, allz+1), axis=1),
            np.concatenate((x, allz+1, allz+1), axis=1), 
            np.concatenate((allz, x, allz+1), axis=1),
            np.concatenate((allz+1, x, allz+1), axis=1),

            # Front (towards camera) - y stays constant
            np.concatenate((allz, allz, x), axis=1), #front right edge
            np.concatenate((allz, allz+1, x), axis=1), #front left edge

            # Back (away from camera)
            np.concatenate((allz+1, allz, x), axis=1), #back right edge
            np.concatenate((allz+1, allz+1, x), axis=1), #back left edge           
        ),
        axis=0)
    return box

# apply 3x4 projection matrix T to nx3 points p1
def applyTransform(T, p1):
    allo = np.ones((p1.shape[0], 1))

    # make homogeneous coordinates by concatenating p1 and allo
    p2 = np.concatenate((p1, allo), axis=1)
    # multiply T*p2.transpose to get homogeneous image coords (then transpose again)
    p2 = np.transpose(p2)
    p2 = np.matmul(T, p2)
    p2 = np.transpose(p2)
    
    # make inhomogeneous coords by dividing with last column 
    u = p2[:,0] / p2[:,2]
    u = np.reshape(u, (-1,1))
    v = p2[:,1] / p2[:,2]
    v = np.reshape(v, (-1,1))

    uv = np.concatenate((u,v), axis =1)

    return uv

# makes rotation and translation matrices 
# t = 3x1 vecctor for the translation; 
# r = 3x1 vector of rotations in radians
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

# camera projection 
# R, t - rotation and translation matrices 
def project(P, K, R, t, image_size=(640, 480)):
    
    # multiply K*R*t use np.matmul twice
    Rt = np.matmul(R,t)
    KRt = np.matmul(K, Rt)

    p = applyTransform(KRt, P)
    p = p.astype(np.uint32)
    
    print('p shape:', p.shape)
    # make image 
    I = np.zeros((image_size[1], image_size[0]))

    # put 255 for points at p_idx    
    I[p[:,1],p[:,0]] = 255
    return p, I


def main():
    n_pts = 1000    #number of points in the cube point cloud
    plot_pts =1     #flag to toggle plotting of the cube points
    # figure out rotation and translation that brings cube into view 
    rot_x = 0         #rotation angle in degrees about the X axis
    rot_y = -90      #rotation angle in degrees about the Y axis
    rot_z = 90    #rotation angle in degrees about the Z axis
    f = 300         # focal length - change to see how projection changes 
    trans_x = 2  # Translation along the X axis
    trans_y = 0  # Translation along the Y axis
    trans_z = 0  # Translation along the Z axis
    image_size = [640,480] #Image size as a string, e.g. 640x480
    scale_factor = 1       # scale factor for the cube
    
    box = get3Dcube(n_pts)
    print(box.shape)
    rot_x = np.deg2rad(rot_x)
    rot_y = np.deg2rad(rot_y)
    rot_z = np.deg2rad(rot_z)

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


main()