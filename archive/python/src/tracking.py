import cv2 
import numpy as np
from helper_funcs import *

class objectTracker():

    def __init__(self):
        # Constan-ts
        self.detect_conf_thres = 0.4
        self.class_conf_thres = 0.25
        self.nms_thres = 0.4

        # Variables
        self.frame_count = 0

        # Video I/O
        self.cap = cv2.VideoCapture('/home/jeric/tracking_ws/source/video1.avi') # Create a VideoCapture object
        # self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"FPS: {self.fps}, Width: {self.fw}, Height: {self.fh}")
        self.out = cv2.VideoWriter(f"/home/jeric/tracking_ws/ouput/tracker_py.mp4",cv2.VideoWriter_fourcc('m','p','4','v'),self.fps,(self.fw,self.fh)) # create writer obj

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

    # handle new tracks and deleted tracks aft a few hits or miss
    def refresh_track(self, current_frame):
        print('Refreshing Track...')

        # add unmatched detections to track
        for detect_id in range(len(self.detect_bboxes)):
            tracker = cv2.legacy.TrackerMOSSE_create()
            success = tracker.init(current_frame, self.detect_bboxes[detect_id])
            tracker_info = np.array([tracker, self.detect_class_ids[detect_id], 1, 0]) # list of [tracker, class_id, no of hits, no of misses]
            self.multi_tracker = np.append(self.multi_tracker, [tracker_info], axis=0)
        

    def track(self, current_frame):
        print('Tracking...')

        # Update the multi-object tracker
        self.get_next_multi_tracker(current_frame)
        
        # Extract class ids from multi_tracker
        self.track_class_ids = self.multi_tracker[:,1]


    # Update the multi-object tracker and return track bboxes in list (for track)
    def get_next_multi_tracker(self, current_frame):
        self.track_bboxes = []
        for track_id in range(len(self.multi_tracker)):
            success, box = self.multi_tracker[track_id][0].update(current_frame)
            self.track_bboxes.append(box)

    
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
        if object_tracker.frame_count == 10:
            # pass through detector
            object_tracker.detect(current_frame)
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