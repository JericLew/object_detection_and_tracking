import cv2 
import numpy as np
from helper_funcs import *

class objectTracker():

    def __init__(self):
        # Constan-ts
        self.dect_conf_thres = 0.4
        self.class_conf_thres = 0.25
        self.max_miss_streak = 5
        self.min_hit_streak = 5

        # Variables
        self.frame_count = 0

        # Video I/O
        self.cap = cv2.VideoCapture('/home/jeric/tracking_ws/video_input/video2.avi') # Create a VideoCapture object
        # self.cap = cv2.VideoCapture(0)  
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"FPS: {self.fps}, Width: {self.fw}, Height: {self.fh}")
        self.out = cv2.VideoWriter(f"/home/jeric/tracking_ws/video_output/output_tracker.mp4",cv2.VideoWriter_fourcc('m','p','4','v'),self.fps,(self.fw,self.fh)) # create writer obj

        # Detector init
        self.net = cv2.dnn.readNet('/home/jeric/yolov5/yolov5s.onnx') # input obj detector network

        # Multi-Tracker init
        self.active_trackers = np.empty((0, 4)) # create tracker list
        self.pending_trackers = np.empty((0, 4)) # create tracker list


    def detect(self, current_frame):
        print('Detecting Objects...')

        # format image in to fit 640x640 and colour
        blob, input_image = format_yolov5(current_frame) 

        # set formatted image into net and pass through net to obtain output
        self.net.setInput(blob)
        predictions = self.net.forward()
        output = predictions[0]

        # unwrap detections into usable format
        boxes, confidences, class_ids = unwrap_detection(input_image,output, self.dect_conf_thres, self.class_conf_thres)

        # NMS to remove dup and overlap
        result_boxes, result_confidences, result_class_ids = nms(boxes, confidences, class_ids)

        return result_boxes, result_confidences, result_class_ids

    def init_track(self, current_frame, result_boxes, result_confidences, result_class_ids):
        print('Init Track...')

        # create track for each detection
        for i in range(len(result_boxes)):
            tracker = cv2.legacy.TrackerKCF.create()
            success = tracker.init(current_frame, result_boxes[i])
            tracker_info = np.array([tracker, result_class_ids[i], 1, 0]) # list of [tracker, class_id, no of hits, no of misses] 
            self.active_trackers = np.append(self.active_trackers, [tracker_info], axis=0)

    # refresh active tracks
    def refresh_active_track(self, current_frame, matches, unmatched_tracks, unmatched_detections, result_boxes, detect_class_ids):
        '''
        TODO improve class handling (might have diff classs and dk which correct)
        '''
        print('Refreshing Active Track...')
        
        # if track and detect match, add 1 to hit streak and update class if needed
        for track_id, detect_id in matches:
            self.active_trackers[track_id][2] += 1
            if self.active_trackers[track_id][1] != detect_class_ids[detect_id]:
                self.active_trackers[track_id][1] = detect_class_ids[detect_id]

    # handle new tracks/pending tracks
    def refresh_pending_track(self, current_frame, matches, unmatched_tracks, unmatched_detections, result_boxes, detect_class_ids):
        # add unmatched detections to track
        for detect_id in unmatched_detections:
            tracker = cv2.legacy.TrackerKCF.create()
            success = tracker.init(current_frame, result_boxes[detect_id])
            tracker_info = np.array([tracker, detect_class_ids[detect_id], 1, 0]) # list of [tracker, class_id, no of hits, no of misses] 
            self.pending_trackers = np.append(self.pending_trackers, [tracker_info], axis=0)

    # handle missed tracks
    def refresh_missed_track(self, unmatched_tracks):
        # +1 miss_streak to missed tracks
        for track_id in unmatched_tracks:
            self.active_trackers[track_id][3] += 1
            if self.active_trackers[track_id][3] >= self.max_miss_streak:
                self.active_trackers[track_id][0] = None

    def track(self, current_frame):
        print('Tracking...')

        # Update the multi-object tracker
        track_boxes = self.update_multi_tracker(current_frame)
        
        # Extract class ids from active_trackers
        class_ids = self.active_trackers[:,1]

        draw_bbox(current_frame, track_boxes, class_ids, tracking=True)

        # Write the frame into the file 'output.avi' 
        self.out.write(current_frame)
        
        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("camera", 1280, 720)
        cv2.imshow("camera", current_frame) # Display image
        cv2.waitKey(1)
    
    # Update the multi-object tracker and return track boxes in list
    def update_trackers(self, current_frame, multi_tracker):
        track_boxes = []
        for tracker in multi_tracker[:, 0]:
            if tracker == None:
                continue
            success, box = tracker.update(current_frame)
            track_boxes.append(box)
        return track_boxes



def main(args=None):
    object_tracker = objectTracker()
    while 1:
        ret, current_frame = object_tracker.cap.read()

        # if no tracks init tracks: do detection and init track
        if not np.any(object_tracker.active_trackers):
                                    
            # Start the timer
            start_time = cv2.getTickCount()

            # pass through detector
            result_boxes, result_confidences, result_class_ids = object_tracker.detect(current_frame)

            # initialise tracking
            object_tracker.init_track(current_frame,result_boxes, result_confidences, result_class_ids)
        
            # Calculate the elapsed timE
            ticks = cv2.getTickCount() - start_time
            elapsed_time = (ticks / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
            print("Time taken for detect and init track:", elapsed_time, "milliseconds")

        # if have init tracks at nth frame: do detection and matching
        elif object_tracker.frame_count % 10 == 0:

            # Start the timer
            start_time = cv2.getTickCount()

            # pass through detector
            result_boxes, result_confidences, result_class_ids = object_tracker.detect(current_frame)

            # update active tracks
            track_boxes = object_tracker.update_multi_tracker(current_frame)
            # handle active tracks matching
            matches, unmatched_tracks, unmatched_detections = hung_algo(track_boxes , result_boxes, object_tracker.active_trackers)
            object_tracker.refresh_active_track(current_frame, matches, unmatched_tracks, unmatched_detections, result_boxes, result_class_ids)

            # update pending tracks
            track_boxes = object_tracker.update_multi_tracker(current_frame)
            # handle pending tracks matching
            matches, unmatched_tracks, unmatched_detections = hung_algo(track_boxes , result_boxes, object_tracker.active_trackers)
            object_tracker.refresh_pending_track(current_frame, matches, unmatched_tracks, unmatched_detections, result_boxes, result_class_ids)

            # handle dead tracks
            object_tracker.refresh_missed_track(unmatched_tracks)

            # Calculate the elapsed timE
            ticks = cv2.getTickCount() - start_time
            elapsed_time = (ticks / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
            print("Time taken for detect and refresh track:", elapsed_time, "milliseconds")

        # if have init tracks and not at nth frame: continue tracking
        else:
            # Start the timer
            start_time = cv2.getTickCount()

            object_tracker.track(current_frame)

            # Calculate the elapsed timE
            ticks = cv2.getTickCount() - start_time
            elapsed_time = (ticks / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
            print("Time taken for continue track:", elapsed_time, "milliseconds")

        object_tracker.frame_count += 1



if __name__ == '__main__':
    main()
