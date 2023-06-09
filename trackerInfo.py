import cv2 
import numpy as np
from helper_funcs import *

class trackerInfo():
    def __init__(self):
        self.tracker = cv2.legacy.TrackerKCF.create()
        self.class_id = -1
        self.hit_streak = 0
        self.miss_streak = 0

    def get_update_bbox(self, current_frame):
        if self.tracker == None:
            return None
        else:
            success, bbox = self.tracker.update(current_frame)
            return success, bbox
        
class multiTracker():
    def __init__(self):
        self.active_trackers = np.empty((0, 4))
        self.pending_trackers = np.empty((0, 4))
    
    def init_track(self):
        pass

    def refresh_track(self):
        pass
    
    def track(self):
        pass
    
    def update_multi(self):
        pass
