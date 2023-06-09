import cv2 
import numpy as np
from helper_funcs import *
'''
TODO
- Fix drifing (readd track)
- Make rematch rate dependant on FPS
- Increase speed
'''
class objectTracker():

    def __init__(self):
        # Constan-ts
        self.detect_conf_thres = 0.4
        self.class_conf_thres = 0.25
        self.nms_thres = 0.4
        self.max_miss_streak = 9 # max number of misses during track refresh
        self.min_hit_streak = 3 # min number of hits during track refresh
        self.rematch_rate = 10 # no of frames per track refresh

        # Variables
        self.frame_count = 0

        # Video I/O
        self.cap = cv2.VideoCapture('/home/jeric/tracking_ws/video_input/video3.avi') # Create a VideoCapture object
        # self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"FPS: {self.fps}, Width: {self.fw}, Height: {self.fh}")
        self.out = cv2.VideoWriter(f"/home/jeric/tracking_ws/video_output/detect_tracker_output.mp4",cv2.VideoWriter_fourcc('m','p','4','v'),self.fps,(self.fw,self.fh)) # create writer obj

        # Detector init
        
        self.net = cv2.dnn.readNet('/home/jeric/tracking_ws/models/yolov5s.onnx') # input obj detector network

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Multi-Tracker init
        self.multi_tracker = np.empty((0, 4)) # create tracker list

        # Data storage
        self.track_bboxes = []
        self.track_class_ids = []
        self.detect_bboxes = []
        self.detect_conf = []
        self.detect_class_ids = []
        self.matches = []
        self.unmatched_tracks = []
        self.unmatched_detections = []


    def detect(self, current_frame):
        print('Detecting Objects...')

        # format image in to fit 640x640 and colour
        blob, input_image = format_yolov5(current_frame) 

        # set formatted image into net and pass through net to obtain output
        self.net.setInput(blob)
        predictions = self.net.forward()
        output = predictions[0]

        # unwrap detections into usable format
        bboxes, confidences, class_ids = unwrap_detection_numpy(input_image,output, self.detect_conf_thres, self.class_conf_thres)

        # NMS to remove dup and overlap
        result_bboxes, result_confidences, result_class_ids = nms(bboxes, confidences, class_ids, self.detect_conf_thres, self.nms_thres)

        self.detect_bboxes = result_bboxes
        self.detect_conf = result_confidences
        self.detect_class_ids = result_class_ids
        
    def association(self, current_frame):
        track_bboxes = self.update_multi_tracker(current_frame)
        self.matches, self.unmatched_tracks, self.unmatched_detections = \
            hung_algo(track_bboxes, self.detect_bboxes) # matches in (track_id, detect_id)

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
                

    def track(self, current_frame):
        print('Tracking...')

        # Update the multi-object tracker
        self.track_bboxes = self.get_next_multi_tracker(current_frame)
        
        # Extract class ids from multi_tracker
        self.track_class_ids = self.multi_tracker[:,1]


    # Update the multi-object tracker and return track bboxes in list
    def update_multi_tracker(self, current_frame):
        track_bboxes = []
        for track_id in range(len(self.multi_tracker)):
            if self.multi_tracker[track_id][0] == None:
                track_bboxes.append(None)
            else:
                success, box = self.multi_tracker[track_id][0].update(current_frame)
                track_bboxes.append(box)
        return track_bboxes

    
    def get_next_multi_tracker(self, current_frame):
        track_bboxes = []
        for track_id in range(len(self.multi_tracker)):
            if self.multi_tracker[track_id][0] == None or self.multi_tracker[track_id][2] < self.min_hit_streak:
                track_bboxes.append(None)
            else:
                success, box = self.multi_tracker[track_id][0].update(current_frame)
                track_bboxes.append(box)
        return track_bboxes

    
    def write_frame(self, current_frame):
        # Write the frame into the file 'output.avi' 
        self.out.write(current_frame)

    
    def display_frame(self, current_frame):
        cv2.imshow("camera", current_frame) # Display image
        cv2.waitKey(1)
        

def main(args=None):
    object_tracker = objectTracker()
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

        draw_bbox(current_frame, object_tracker.track_bboxes, object_tracker.track_class_ids, tracking=True)
        object_tracker.write_frame(current_frame)
        object_tracker.display_frame(current_frame)
        object_tracker.frame_count += 1

        # Calculate the elapsed time
        ticks = cv2.getTickCount() - start_time
        elapsed_time = (ticks / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
        print("Time taken:", elapsed_time, "milliseconds\n")


if __name__ == '__main__':
    # Start the timer
    total_start_time = cv2.getTickCount()

    main()
    
    # Calculate the elapsed time
    total_ticks = cv2.getTickCount() - total_start_time
    total_elapsed_time = (total_ticks / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
    print("Total Time taken:", total_elapsed_time, "milliseconds\n")