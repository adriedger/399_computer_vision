import cv2
import time, argparse, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def getImages(n_frames, diff=0, src_dir=''):
    img_seq = []
    if src_dir:
        print('Capturing images from {}'.format(src_dir))
        cap = cv2.VideoCapture(os.path.join(src_dir, 'image_%d.bmp'))
    else:
        print('Capturing images from camera')
        cap = cv2.VideoCapture(0)
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print('Capture of Image {} was unsuccessful'.format(i + 1))
            break
        img_gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_seq.append(img_gs)
        cv2.imshow('getImages', frame)
        time.sleep(diff)
        cv2.waitKey(1)
    cv2.destroyWindow('getImages')
    return img_seq


def getDifference(img_seq, threshold):
    n_frames = len(img_seq)
    img_seq_diff = []
    for i in range(n_frames - 1):
        frame_diff = img_seq[i + 1].astype(np.float64) - img_seq[i].astype(np.float64)
        frame_diff[np.abs(frame_diff) < threshold] = 0
        cv2.imshow('getDifference', np.abs(frame_diff).astype(np.uint8))
        img_seq_diff.append(frame_diff)
        cv2.waitKey(1)
    cv2.destroyWindow('getDifference')
    return img_seq_diff

def getGridPoints(img_width,img_height):
    _X = np.asarray(range(0, img_width - 1, block_size)) + block_size // 2
    _Y = np.asarray(range(0, img_height - 1, block_size)) + block_size // 2

    [X, Y] = np.meshgrid(_X, _Y)
    return X,Y

# using open cv function 
def computeOpticalFlow_opencv(img, img_old, threshold, block_size,lk_params):
    
    img_height, img_width = img_seq[0].shape[:2]
    X,Y = getGridPoints(img_width,img_height)
    n,m = X.shape
    
    
    # force difference to zero 
    frame_diff = img.astype(np.float64) - img_old.astype(np.float64)
    sel = np.abs(frame_diff) < threshold
    img[sel] = img_old[sel]
    
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    p0=np.concatenate((X,Y),axis=1)
    p0 = p0.astype(np.float32)
    p0 = p0.reshape(-1,1,2)
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(img_old, img, p0, None, **lk_params)
   
    U = (p1[:, 0, 0] - p0[:, 0, 0])
    V = (p1[:, 0, 1] - p0[:, 0, 1])
    # thid doesn't work very well
    st = st.squeeze()
    # st based on differences  
    frame_diff = img.astype(np.float64) - img_old.astype(np.float64)
    st = np.abs(frame_diff) < threshold
    U[st] = 0
    V[st] = 0    
    
    
    #U = U.reshape((n, m))
    #V = V.reshape((n, m))    
    return X,Y,U,V


def computeOpticalFlow(img, img_diff, block_size, grad_type=0):
    img_height, img_width = img.shape[:2]

    X,Y = getGridPoints(img_width,img_height)

    if grad_type == 0:
        [img_grad_x, img_grad_y] = np.gradient(img)
    else:
        img_grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        img_grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    n, _ = X.shape
    _, m = Y.shape

    U = np.zeros((n, m))
    V = np.zeros((n, m))

    for ix in range(n):
        for iy in range(m):
            block_start_x = int(X[ix, iy] - block_size // 2)
            block_start_y = int(Y[ix, iy] - block_size // 2)
            block_end_x = block_start_x + block_size
            block_end_y = block_start_y + block_size

            block_grad_x = img_grad_x[block_start_y:block_end_y, block_start_x:block_end_x]
            block_grad_y = img_grad_y[block_start_y:block_end_y, block_start_x:block_end_x]
            block_diff = img_diff[block_start_y:block_end_y, block_start_x:block_end_x]

            block_grad_x_vec = np.reshape(block_grad_x, (-1, 1))
            block_grad_y_vec = np.reshape(block_grad_y, (-1, 1))
            block_grad = np.concatenate((block_grad_x_vec, block_grad_y_vec), axis=1)
            block_diff_vec = np.reshape(block_diff, (-1, 1))

            res = -np.dot(np.linalg.pinv(block_grad), block_diff_vec)

            x_motion, y_motion = res.squeeze()
            U[ix, iy] = x_motion
            V[ix, iy] = y_motion
    return X, Y, U, V
    
#--------MAIN---------

img_seq = getImages(30, 0, src_dir='Arm32im')
n_images = len(img_seq)
img_seq_diff = getDifference(img_seq, 30)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# visualize differences 
for i in range(n_images-1):
    plt.subplot(121)
    plt.imshow(img_seq[i+1],cmap='gray')
    plt.xticks([]), plt.yticks([]) 
    plt.subplot(122)
    plt.imshow(np.abs(img_seq_diff[i]).astype(np.uint8),cmap='gray')
    plt.xticks([]), plt.yticks([])        
    plt.pause(0.5) 



# compute and visualize optic flow using matplotlib.animation 
block_size=10
fig, ax = plt.subplots()
quiver_scale = 0.005 #for your method 
#quiver_scale = 0.05 # for opencv method  
def update(i):
    img = img_seq[i + 1]
    X, Y, U, V = computeOpticalFlow(img, img_seq_diff[i], block_size, 0)
    #X, Y, U, V = computeOpticalFlow_opencv(img, img_seq[i],  30, block_size,lk_params)
    
    ax.imshow(img, cmap='gray')
    ax.hold(1) 
    ax.quiver(X, Y, U, V, units='dots', scale=quiver_scale, color='blue')   
    ax.hold(0)

ani = anim.FuncAnimation(fig, update, frames = n_images-1, interval=1, repeat=False)
plt.show()

save_animation = 0
if save_animation:
    print('Saving animation...')
    FFMpegWriter = anim.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    ani.save('opt_flow.mp4', writer=writer)

#------------------
# use open cv dense optic flow cv2.calcOpticalFlowFarneback
'''
hsv = np.zeros((img_seq[0].shape[0],img_seq[0].shape[1],3))
hsv[...,1] = 255
prvs = img_seq[0]
for i in range(n_images-1):
    next = img_seq[i+1]

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',rgb)
    cv2.waitKey(0) 
    prvs = next

cv2.destroyAllWindows()

#-----------------------------------------
# use opencv cv2.calcOpticalFlowPyrLK
# https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


# using Shi Tomasi features
p0 = cv2.goodFeaturesToTrack(img_seq[0], mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(img_seq[0])
# Create some random colors
color = np.random.randint(0,255,(100,3))
img_old = img_seq[0]

for i in range(n_images-1):
    
    img_new = img_seq[i+1]
    frame = img_new
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(img_old, img_new, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    cv2.waitKey(0) 

    # Now update the previous frame and previous points
    img_old = img_new.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
'''