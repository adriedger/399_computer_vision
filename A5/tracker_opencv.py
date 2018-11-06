class Tracker:
    def __init__(self):
        self.initial_corners = None
        self.current_corners = None

    def initialize(self, img, corners):
        # initialize your tracker with the first frame from the sequence and
        # the corresponding corners from the ground truth
        # this function does not return anything
        self.initial_corners = corners
        # ...

    def update(self, img):
        # update your tracker with the current image and return the current corners
        # ...
        return self.current_corners
       
