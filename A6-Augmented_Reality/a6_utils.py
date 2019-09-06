import cv2
import os, sys
import numpy as np


def getTrackingObject(cap, col=(0, 0, 255), title=None, line_thickness=2):
    if title is None:
        title = 'Select the object to track'
    cv2.namedWindow(title)

    pts = []
    hover_pt = [None]

    def drawLines(img, _pts, _hover_pt=None):
        if len(_pts) == 0:
            # cv2.imshow(title, img)
            return
        for i in range(len(_pts) - 1):
            cv2.line(img, _pts[i], _pts[i + 1], col, line_thickness)
        if _hover_pt is None:
            return
        cv2.line(img, _pts[-1], _hover_pt, col, line_thickness)
        if len(_pts) == 3:
            cv2.line(img, _pts[0], _hover_pt, col, line_thickness)
        elif len(_pts) == 4:
            cv2.line(img, _pts[0], _pts[-1], col, line_thickness)
        # cv2.imshow(title, img)

    def mouseHandler(event, x, y, flags=None, param=None):
        if len(pts) >= 4:
            return
        # if event != cv2.EVENT_MOUSEMOVE:
        #     hover_pt = None

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
            # temp_img = annotated_img.copy()

        elif event == cv2.EVENT_RBUTTONUP:
            pass
        elif event == cv2.EVENT_MBUTTONDOWN:
            pass
        elif event == cv2.EVENT_MOUSEMOVE:
            hover_pt[0] = (x, y)

    ret, img = cap.read()
    if not ret:
        raise IOError("Frame could not be read")

    cv2.setMouseCallback(title, mouseHandler)

    while len(pts) < 4:
        key = cv2.waitKey(1)
        if key == 27:
            return None
        ret, img = cap.read()
        if not ret:
            raise IOError("Frame could not be read")

        # print('hover_pt: {}'.format(hover_pt))

        drawLines(img, pts, hover_pt[0])
        cv2.imshow(title, img)

    cv2.waitKey(250)
    cv2.destroyWindow(title)
    corners = np.array(pts).T
    return corners


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

def rearrangeCorners(corners):
    rect_x, rect_y, rect_w, rect_h = getBestFitRectangle(corners)
    rect_pts = np.float32(
        [[rect_x, rect_y],
         [rect_x + rect_w - 1, rect_y],
         [rect_x + rect_w - 1, rect_y + rect_h - 1],
         [rect_x, rect_y + rect_h - 1]]
    )
    out_corners = []
    for i in range(4):
        rt_x, rt_y = rect_pts[i, :]
        min_dist = np.inf
        min_dist_pt = None
        for j in range(4):
            cr_x, cr_y = corners[:, j]
            dist = (rt_x-cr_x)**2 + (rt_y-cr_y)**2
            if dist<min_dist:
                min_dist = dist
                min_dist_pt = [cr_x, cr_y]
        out_corners.append(min_dist_pt)
    out_corners = np.array(out_corners).T
    return out_corners


def getBestFitRectangle(corners):
    centroid = np.mean(corners, axis=1)
    mean_half_size = np.mean(np.abs(corners - centroid.reshape((2, 1))), axis=1)

    top_left = np.squeeze(centroid - mean_half_size).astype(np.int32)
    rect_size = np.squeeze(2 * mean_half_size).astype(np.int32)

    return top_left[0], top_left[1], rect_size[0], rect_size[1]

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

