import sys
import cv2
import numpy as np


class Tracker:
    def __init__(self, parser=None):
        # Parameters
        self.tracker_types = {
            0: ('MEDIANFLOW', cv2.TrackerMedianFlow_create),
            1: ('KCF', cv2.TrackerKCF_create),
            2: ('BOOSTING', cv2.TrackerBoosting_create),
            3: ('MIL', cv2.TrackerMIL_create),
            4: ('TLD', cv2.TrackerTLD_create),
            5: ('GOTURN', cv2.TrackerGOTURN_create)
        }

        self.tracker_type_id = 0
        self.gauss_kernel_size = 3
        self.patch_size = 50
        self.update_template = 0
        self.show_patches = 0
        self._pause = 0

        if parser is not None:
            parser.add_argument("--tracker_type_id", type=int, default=self.tracker_type_id)
            parser.add_argument("--gauss_kernel_size", type=int, default=self.gauss_kernel_size)
            parser.add_argument("--patch_size", type=int, default=self.patch_size)
            parser.add_argument("--show_patches", type=int, default=self.show_patches)

        self.iniit_corners = None
        self.curr_rect = self.curr_corners = None
        self.init_img = self.curr_img = None
        self.trackers = []
        self.curr_corners = np.zeros((2, 4), dtype=np.float64)

    def initialize(self, img, corners, args=None):
        if args is not None:
            self.tracker_type_id = args.tracker_type_id
            self.gauss_kernel_size = args.gauss_kernel_size
            self.patch_size = args.patch_size
            self.show_patches = args.show_patches

        tracker_type, tracker_create = self.tracker_types[self.tracker_type_id]
        print('Using {} tracker'.format(tracker_type))

        self.init_img = img
        self.iniit_corners = corners

        # if len(self.init_img.shape) == 3:
        #     self.init_img = cv2.cvtColor(self.init_img, cv2.COLOR_BGR2GRAY)

        if self.gauss_kernel_size:
            self.init_img = cv2.GaussianBlur(self.init_img, (self.gauss_kernel_size, self.gauss_kernel_size), 0)

        half_size = self.patch_size / 2.0
        for corner_id in range(4):

            cv_tracker = tracker_create()

            cx, cy = corners[:, corner_id]
            xmin = int(cx - half_size)
            ymin = int(cy - half_size)
            roi = (xmin, ymin, self.patch_size, self.patch_size)
            ok = cv_tracker.init(self.init_img, roi)
            if not ok:
                raise SystemError('Tracker {} initialization was unsuccessful'.format(corner_id + 1))
            self.trackers.append(cv_tracker)

    def update(self, img):
        self.curr_img = img

        # if len(self.curr_img.shape) == 3:
        #     self.curr_img = cv2.cvtColor(self.curr_img, cv2.COLOR_BGR2GRAY)

        if self.gauss_kernel_size:
            self.curr_img = cv2.GaussianBlur(self.curr_img, (self.gauss_kernel_size, self.gauss_kernel_size), 0)

        for corner_id in range(4):
            ok, bbox = self.trackers[corner_id].update(self.curr_img)
            if not ok:
                # print('bbox: {}'.format(bbox))
                raise SystemError('Tracker {} update was unsuccessful'.format(corner_id + 1))

            xmin, ymin, width, height = bbox
            cx = xmin + width / 2.0
            cy = ymin + height / 2.0
            self.curr_corners[:, corner_id] = (cx, cy)

            if self.show_patches:
                cv2.rectangle(self.curr_img, (int(xmin), int(ymin)), (int(xmin + width), int(ymin + height)),
                              (0, 0, 255), 2)

        if self.show_patches:
            cv2.imshow('Patches', self.curr_img)
            k = cv2.waitKey(1 - self._pause)
            if k == 27:
                sys.exit(0)
            elif k == 32:
                self._pause = 1 - self._pause

        if self.update_template:
            pass

        return self.curr_corners
