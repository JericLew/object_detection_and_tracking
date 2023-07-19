import cv2 
import numpy as np
from helper_funcs import *
import argparse
import os

# Create the argument parser
parser = argparse.ArgumentParser(description='Detection engine written in python with opencv')

# Add the input and output file arguments
parser.add_argument('tracking_ws_path', help='/path/to/tracking_ws/')
parser.add_argument('input_video_path', help='/path/to/video/file/video.xxx')


# Parse the arguments
args = parser.parse_args()
input_video_path = args.input_video_path
tracking_ws_path = args.tracking_ws_path
video_name = input_video_path.split('/')[-1]
video_name_no_ext = video_name.split('.')[0]
output_video_path = os.path.join(tracking_ws_path,'output',video_name_no_ext+'_detect_py.mp4')
print(output_video_path)

class objectDetector():

    def __init__(self):
        # Constan-ts
        self.detect_conf_thres = 0.4
        self.class_conf_thres = 0.25
        self.nms_thres = 0.4

        # Variables
        self.frame_count = 0

        # Video I/O
        self.cap = cv2.VideoCapture(input_video_path) # Create a VideoCapture object
        # self.cap = cv2.VideoCapture(0)  
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"FPS: {self.fps}, Width: {self.fw}, Height: {self.fh}")
        self.out = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc('m','p','4','v'),self.fps,(self.fw,self.fh)) # create writer obj

        # Detector init
        self.net = cv2.dnn.readNet('/home/jeric/tracking_ws/models/last_exp2.onnx') # input obj detector network

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Data storage
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
    

def main(args=None):
    object_detector = objectDetector()
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("camera", 1280, 720)

    while 1:
        # Start the timer
        start_time = cv2.getTickCount()

        ret, current_frame = object_detector.cap.read()

        if not ret:
            break
             
        # pass through detector
        object_detector.detect(current_frame)
        draw_bbox(current_frame, object_detector.detect_bboxes, object_detector.detect_class_ids, tracking=False)

        # Write the frame into the file 'output.avi' 
        object_detector.out.write(current_frame)
        
        cv2.imshow("camera", current_frame) # Display image
        cv2.waitKey(1)
        
        # Calculate the elapsed time
        ticks = cv2.getTickCount() - start_time
        elapsed_time = (ticks / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
        print("Time taken for Detection:", elapsed_time, "milliseconds\n")


if __name__ == '__main__':
    main()
