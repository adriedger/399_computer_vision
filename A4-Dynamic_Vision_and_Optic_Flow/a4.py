import cv2
import time, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# question 1.a# capture imges from a src_dir or camera
def getImages(n_frames, diff=0, src_dir=''):
    
    # you will store images in a list img_seq 
    # ... 
    # you can use cv2.VideoCapture for both cases 
    # initialize the capture - two cases 
    if src_dir:
        print('Capturing images from {}'.format(src_dir))
        # you can assume all images have a standard name image_%d.bmp
        # and form a joint path with all images as filename
        # os.path.join(src_dir, 'image_%d.bmp')
        # ... 
        
    else:
        print('Capturing images from camera')
        # ...
    
    # capture n_frames from your initialized capture object     
    for i in range(n_frames):
        # read one frame
        # ..
        # test if succesful 
        # ..
        # convert to grayscale 
        # ...
        # append to img_seq 
        # .. 
        
        # wait for diff until next is captured
        # can use time.sleep
        
    return img_seq

# question 1. b. 
# computes differences of any two adjacebt frames in img_seq 
# sets pixels in the difference images to 0 if smaller than threshold
def getDifference(img_seq, threshold):
    # you can store the image differences in a list img_seq_diff
    # ...
    for i in range(n_frames - 1):
        # compute difference (make sure you work with floats) 
        # ...
        # set pixels < threshold pixels to zero 
        # ...
        # store image difference in img_seq_diff
        
    return img_seq_diff

# computes point on a grid of of block_size for an image of size [img_width,img_height]
def getGridPoints(img_width,img_height,block_size):
    _X = np.asarray(range(0, img_width - 1, block_size)) + block_size // 2
    _Y = np.asarray(range(0, img_height - 1, block_size)) + block_size // 2

    [X, Y] = np.meshgrid(_X, _Y)
    return X,Y


# 2.b. optic flow using open cv function 
def computeOpticalFlow_opencv(img, img_old, block_size,lk_params):
    
    # get points for which we calculate optic flow 
    img_height, img_width = img.shape[:2]
    X,Y = getGridPoints(img_width,img_height,block_size)
    n,m = X.shape
    
    # convert to form required by the opencv function 
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    p0=np.concatenate((X,Y),axis=1)
    p0 = p0.astype(np.float32)
    p0 = p0.reshape(-1,1,2)
    
    # call opencv function cv2.calcOpticalFlowPyrLK
    # ... 
    # compute U, V optic flow vectors from p1 (new position) and p0 (old position) 
    # could also set to zero components for which st == 0
    # ..
    # ..
    return X,Y,U,V

# 2.a your function for computing optic flow 
def computeOpticalFlow(img, img_diff, block_size, grad_type=0):
    
    # get points for which we calculate optic flow  
    img_height, img_width = img.shape[:2]
    X,Y = getGridPoints(img_width,img_height)

    # compute image gradient img_grad_x, img_grad_y
    # ...
    
    # initialize U, V and n x m matrices where  
    # n,m is the dimension of X (no points as X.shape)
    # ..
    # ..

    for ix in range(n):
        for iy in range(m):
            # extract the blocks of information 
            # for a block of dimension block_size around points in X,Y 
            # make sure X,Y are int coords
            # from img_grad_x
            # ...
            # from img_grad_y
            # ...
            # from img_diff
            # ...
            
            # rearange the blocks are column vectors to make the two matrices in equation 
            # M * res = block_diff_column
            # can use np.reshape
            # grad_x_column = ...
            # grad_y_ colum = ...
            # M has the two column vectors as columns (can use np.concatenate) 
            # block_diff_column = ...
        
            # solve the linesr equation system 
            # can use direct method : dot product between pseudoinverse and block_diff_column
            # res = -np.dot(np.linalg.pinv(M), block_diff_column)
            # or the least square solver 
            
            # extract motion vectors from res and store them in the flow variable U, V 
            # U[ix, iy] = ...
            # V[ix, iy] = ...
    return X, Y, U, V
    
#--------MAIN---------

img_seq = getImages(30, 0, src_dir='Arm32im')
n_images = len(img_seq)
img_seq_diff = getDifference(img_seq, 30)
    
# visualize differences 
for i in range(n_images-1):
    plt.subplot(121)
    plt.imshow(img_seq[i+1],cmap='gray')
    plt.xticks([]), plt.yticks([]) 
    plt.subplot(122)
    plt.imshow(np.abs(img_seq_diff[i]).astype(np.uint8),cmap='gray')
    plt.xticks([]), plt.yticks([])        
    plt.pause(0.5) 

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# compute and visualize optic flow using matplotlib.animation 
block_size=10
quiver_scale = 0.005 #for your method 
#quiver_scale = 0.05 # for opencv method  
fig, ax = plt.subplots()
def update(i):
    img = img_seq[i + 1]
    X, Y, U, V = computeOpticalFlow(img, img_seq_diff[i], block_size, 0)
    #X, Y, U, V = computeOpticalFlow_opencv(img, img_seq[i], block_size,lk_params)
    
    ax.imshow(img, cmap='gray')
    ax.hold(1) 
    ax.quiver(X, Y, U, V, units='xy', scale=quiver_scale, color='blue')   
    ax.hold(0)

ani = anim.FuncAnimation(fig, update, frames = n_images-1, interval=1, repeat=False)
plt.show()

# same animation as mp4 file
save_animation = 0
if save_animation:
    print('Saving animation...')
    FFMpegWriter = anim.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    ani.save('opt_flow.mp4', writer=writer)
