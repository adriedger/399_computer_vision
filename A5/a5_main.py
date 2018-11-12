import os
import sys
import cv2
import numpy as np
import math
import time

from tracker_opencv import Tracker
#from tracker_lk_opencv import Tracker


def readTrackingData(filename):
    if not os.path.isfile(filename):
        print("Tracking data file not found:\n{}".format(filename))
        sys.exit()

    data_file = open(filename, 'r')
    lines = data_file.readlines()
    no_of_lines = len(lines)
    data_array = np.zeros((no_of_lines, 8))
    line_id = 0
    for line in lines:
        words = line.split()
        if len(words) != 8:
            msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
            raise SyntaxError(msg)
        coordinates = []
        for word in words:
            coordinates.append(float(word))
        data_array[line_id, :] = coordinates
        line_id += 1
    data_file.close()
    return data_array


def writeCorners(file_id, corners):
    # write the given corners to the file
    corner_str = ''
    for i in range(4):
        corner_str = corner_str + '{:5.2f}\t{:5.2f}\t'.format(corners[0, i], corners[1, i])
    file_id.write(corner_str + '\n')


def drawRegion(img, corners, color, thickness=1):
    # draw the bounding box specified by the given corners
    for i in range(4):
        p1 = (int(corners[0, i]), int(corners[1, i]))
        p2 = (int(corners[0, (i + 1) % 4]), int(corners[1, (i + 1) % 4]))
        cv2.line(img, p1, p2, color, thickness)


#-------------------------
# MAIN 

sequences = ['bookI', 'bookII', 'bookIII', 'bus', 'cereal']

# these are some params to set up for the main file 
seq_id = 4
write_stats_to_file = 1
show_tracking_output = 1

# initialize names for input files (sequence and ground truth) 
# and output file name for results 
seq_name = sequences[seq_id]
print('seq_id: {}'.format(seq_id))
print('seq_name: {}'.format(seq_name))

src_fname = 'data/'+seq_name + '/img%03d.jpg'
ground_truth_fname = 'data/ground_truth/'+ seq_name + '.txt'
result_fname = seq_name + '_res.txt'
result_file = open(result_fname, 'w')

cap = cv2.VideoCapture()
if not cap.open(src_fname):
    print('The video file {} could not be opened'.format(src_fname))
    sys.exit()

# thickness of the bounding box lines drawn on the image
thickness = 2
# ground truth location drawn in green
ground_truth_color = (0, 255, 0)
# tracker location drawn in red
result_color = (0, 0, 255)

# read the ground truth
ground_truth = readTrackingData(ground_truth_fname)
no_of_frames = ground_truth.shape[0]

print('no_of_frames: {}'.format(no_of_frames))

ret, init_img = cap.read()
if not ret:
    print("Initial frame could not be read")
    sys.exit(0)

# extract the true corners in the first frame and place them into a 2x4 array
init_corners = [ground_truth[0, 0:2].tolist(),
                ground_truth[0, 2:4].tolist(),
                ground_truth[0, 4:6].tolist(),
                ground_truth[0, 6:8].tolist()]
init_corners = np.array(init_corners).T
# write the initial corners to the result file
writeCorners(result_file, init_corners)

# initialize the tracker 
tracker = Tracker()

# initialize tracker with the first frame and the initial corners
tracker.initialize(init_img, init_corners)
window_name = 'Tracking Result'

if show_tracking_output:
    # window for displaying the tracking result
    cv2.namedWindow(window_name)

# lists for accumulating the tracking error and fps for all the frames
tracking_errors = []
tracking_fps = []

mean_fps = 0
mean_error = 0

# main loop over the frames ; update tracker at each frame 
for frame_id in range(1, no_of_frames):
    ret, src_img = cap.read()
    if not ret:
        print("Frame {} could not be read".format(frame_id))
        break
    actual_corners = [ground_truth[frame_id, 0:2].tolist(),
                      ground_truth[frame_id, 2:4].tolist(),
                      ground_truth[frame_id, 4:6].tolist(),
                      ground_truth[frame_id, 6:8].tolist()]
    actual_corners = np.array(actual_corners).T

    start_time = time.clock()
    # update the tracker with the current frame
    tracker_corners = tracker.update(src_img)
    end_time = time.clock()

    # write the current tracker location to the result text file
    writeCorners(result_file, tracker_corners)

    # compute the tracking fps
    if (end_time - start_time) == 0:
        blah = 1
    else:
        blah = end_time -start_time
    current_fps = 1.0 / blah
    mean_fps += (current_fps-mean_fps)/frame_id


    # compute the tracking error
    current_error = math.sqrt(np.sum(np.square(actual_corners - tracker_corners)) / 4)
    mean_error += (current_error - mean_error) / frame_id

    if show_tracking_output:
        # draw the ground truth location
        drawRegion(src_img, actual_corners, ground_truth_color, thickness)
        # draw the tracker location
        drawRegion(src_img, tracker_corners, result_color, thickness)
        # write statistics (error and fps) to the image
        cv2.putText(src_img, "frame {:d} fps: {:5.2f}({:5.2f}) error: {:5.2f}({:5.2f})".format(
            frame_id, current_fps, mean_fps, current_error, mean_error), (5, 15),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
        # display the image
        cv2.imshow(window_name, src_img)

        if cv2.waitKey(1) == 27:
            break
            # print 'curr_error: ', curr_error

print('mean_error: {}'.format(mean_error))
print('mean_fps: {}'.format(mean_fps))

result_file.close()

if write_stats_to_file:
    fout = open("tracking_stats.txt", "a")
    fout.write('{:s}\t{:d}\t{:12.6f}\t{:12.6f}\n'.format(sys.argv[0], seq_id, mean_error, mean_fps))
    fout.close()
