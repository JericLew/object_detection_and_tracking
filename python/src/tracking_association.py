'''
tracking script for python using correlation filters

NOTE:
Python version does not include frame shrinking and updating bbox size for track
and only supports MOSSE

Arguements are /path/to/tracking_ws and /path/to/video/file
You can change the model used and class list in the init of objectDetector class
You can change tracking settings below in constants
'''

import cv2
import os
import argparse
import numpy as np
from helper_funcs import *
from detection import *

class objectTracker(objectDetector):

    def __init__(self, tracking_ws_path, input_video_path):
        super().__init__(tracking_ws_path, input_video_path)
        # Handle arguements
        video_name = input_video_path.split('/')[-1]
        video_name_no_ext = video_name.split('.')[0]
        output_video_path = os.path.join(tracking_ws_path,'output',video_name_no_ext+'_track_py.mp4')

        # Constants
        self.max_miss_streak = 9 # max number of misses during track refresh
        self.min_hit_streak = 3 # min number of hits during track refresh
        self.rematch_rate = 1 # no of frames per track refresh

        # Video I/O
        self.out = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc('m','p','4','v'),self.fps,(self.fw,self.fh)) # create writer obj

        # Multi-Tracker init
        self.multi_tracker = np.empty((0, 4)) # create tracker list

        # Data storage
        self.track_bboxes = []
        self.track_class_ids = []
        self.matches = []
        self.unmatched_tracks = []
        self.unmatched_detections = []

    def association(self, current_frame):
        self.track_bboxes = self.update_multi_tracker(current_frame)
        self.matches, self.unmatched_tracks, self.unmatched_detections = \
            hung_algo(self.track_bboxes, self.detect_bboxes) # matches in (track_id, detect_id)

    # handle new tracks and deleted tracks aft a few hits or miss
    def refresh_track(self, current_frame):
        print('Refreshing Track...')

        # if track and detect match, add 1 to hit streak and update class if needed
        for track_id, detect_id in self.matches:
            if self.multi_tracker[track_id][0] == None:
                continue
            self.multi_tracker[track_id][2] += 1 # +1 to hit_streak
            self.multi_tracker[track_id][3] = 0 # reset miss_streak
            if self.multi_tracker[track_id][1] != self.detect_class_ids[detect_id]:
                self.multi_tracker[track_id][1] = self.detect_class_ids[detect_id]

        # add unmatched detections to track
        for detect_id in self.unmatched_detections:
            tracker = cv2.legacy.TrackerMOSSE_create()
            success = tracker.init(current_frame, self.detect_bboxes[detect_id])
            tracker_info = np.array([tracker, self.detect_class_ids[detect_id], 1, 0]) # list of [tracker, class_id, no of hits, no of misses]
            self.multi_tracker = np.append(self.multi_tracker, [tracker_info], axis=0)
        
        # +1 miss_streak to missed tracks
        for track_id in self.unmatched_tracks:
            if self.multi_tracker[track_id][0] == None:
                continue
            # self.multi_tracker[track_id][2] = 0 # reset hit streak
            self.multi_tracker[track_id][3] += 1 # + 1 to miss streak
            if self.multi_tracker[track_id][3] >= self.max_miss_streak: # if miss streak larger or equal to allowed
                self.multi_tracker[track_id][0] = None # make it none if it was already active (so numbers wont jump)  
                
        self.track_class_ids = self.multi_tracker[:,1]           
                

    def track(self, current_frame):
        print('Tracking...')

        # Update the multi-object tracker
        self.get_next_multi_tracker(current_frame)
        
        # Extract class ids from multi_tracker
        self.track_class_ids = self.multi_tracker[:,1]


    # Update the multi-object tracker and return track bboxes in list
    def update_multi_tracker(self, current_frame):
        self.track_bboxes = []
        for track_id in range(len(self.multi_tracker)):
            if self.multi_tracker[track_id][0] == None:
                self.track_bboxes.append(None)
            else:
                success, box = self.multi_tracker[track_id][0].update(current_frame)
                self.track_bboxes.append(box)
        return self.track_bboxes

    
    def get_next_multi_tracker(self, current_frame):
        self.track_bboxes = []
        for track_id in range(len(self.multi_tracker)):
            if self.multi_tracker[track_id][0] == None or self.multi_tracker[track_id][2] < self.min_hit_streak:
                self.track_bboxes.append(None)
            else:
                success, box = self.multi_tracker[track_id][0].update(current_frame)
                self.track_bboxes.append(box)
        return self.track_bboxes

def main(tracking_ws_path, input_video_path):
    object_tracker = objectTracker(tracking_ws_path, input_video_path)
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("camera", 1280, 720)
    cv2.cuda.setDevice(0) 

    while 1:
        # Start the timer
        start_time = cv2.getTickCount()

        ret, current_frame = object_tracker.cap.read()

        if not ret:
            print("Nothing Read")
            break

        # if have init tracks at nth frame: do detection and matching
        if object_tracker.frame_count % object_tracker.rematch_rate == 0:
            # pass through detector
            object_tracker.detect(current_frame)
        
            # do association with updated tracks and detections
            object_tracker.association(current_frame)

            # handle matches, unmatched tracks and unmatched detections
            object_tracker.refresh_track(current_frame)

        # if have init tracks and not at nth frame: continue tracking
        else:
            # continue tracking with updated track
            object_tracker.track(current_frame)

        draw_bbox(current_frame, object_tracker.track_bboxes, object_tracker.track_class_ids, object_tracker.class_list, tracking=True)
        object_tracker.out.write(current_frame)
        cv2.imshow("camera", current_frame)
        cv2.waitKey(1)

        object_tracker.frame_count += 1

        # Calculate the elapsed time
        ticks = cv2.getTickCount() - start_time
        elapsed_time = (ticks / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
        print("Time taken:", elapsed_time, "milliseconds\n")


if __name__ == '__main__':
    # Start the timer
    total_start_time = cv2.getTickCount()

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Detection engine written in python with opencv')

    # Add the input and output file arguments
    parser.add_argument('tracking_ws_path', help='/path/to/tracking_ws/')
    parser.add_argument('input_video_path', help='/path/to/video/file/video.xxx')

    # Parse the arguments
    args = parser.parse_args()
    input_video_path = args.input_video_path
    tracking_ws_path = args.tracking_ws_path

    main(tracking_ws_path, input_video_path)
    
    # Calculate the elapsed time
    total_ticks = cv2.getTickCount() - total_start_time
    total_elapsed_time = (total_ticks / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
    print("Total Time taken:", total_elapsed_time, "milliseconds\n")