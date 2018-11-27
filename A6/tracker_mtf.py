import pyMTF
import numpy as np


class Tracker:
    def __init__(self, parser=None):
        self.tracker_id = None
        self.curr_corners = np.zeros((2, 4), dtype=np.float64)
        self.mtf_config_root_dir = 'H:/UofA/MSc/Code/TrackingFramework/C++/MTF/Config'
        self.mtf_cmd_args = ''

        if parser is not None:
            parser.add_argument("--mtf_config_root_dir", type=str, default=self.mtf_config_root_dir,
                                help='location of MTF config files')
            parser.add_argument("--mtf_cmd_args", type=str, default=self.mtf_cmd_args,
                                help='comma separated list of pairwise arguments for MTF')

    def initialize(self, img, corners, args=None):
        if args is not None:
            self.mtf_config_root_dir = args.mtf_config_root_dir
            self.mtf_cmd_args = args.mtf_cmd_args

        self.mtf_cmd_args = self.mtf_cmd_args.replace(',', ' ')

        self.tracker_id = pyMTF.create(img.astype(np.uint8), corners.astype(np.float64),
                                       self.mtf_config_root_dir, self.mtf_cmd_args)
        if not self.tracker_id:
            raise SystemError('Tracker initialization was unsuccessful')

    def update(self, img):
        success = pyMTF.getRegion(img.astype(np.uint8), self.curr_corners, self.tracker_id)
        if not success:
            raise SystemError('Tracker update was unsuccessful')
        return self.curr_corners

    def close(self):
        success = pyMTF.remove(self.tracker_id)
        if not success:
            raise SystemError('Tracker removal was unsuccessful')
