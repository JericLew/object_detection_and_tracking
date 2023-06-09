import cv2 
import numpy as np

class objectTracker():
    def __init__(self):
        # Constants
        self.dect_conf_thres = 0.4
        self.class_conf_thres = 0.25

        # Variables
        self.frame_count = 0

        # Video I/O
        self.cap = cv2.VideoCapture('/home/jeric/tracking_ws/video_input/video4.avi') # Create a VideoCapture object
        # self.cap = cv2.VideoCapture(0)  
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(self.fps)
        print(self.fw)
        print(self.fh)
        self.out = cv2.VideoWriter(f"/home/jeric/tracking_ws/video_output/tracker_output.mp4",cv2.VideoWriter_fourcc('m','p','4','v'),self.fps,(self.fw,self.fh)) # create writer obj

        # Detector and Tracker init
        self.net = cv2.dnn.readNet('/home/jeric/yolov5/yolov5s.onnx') # input obj detector network
        self.trackers = cv2.legacy.MultiTracker.create() # create trackers
        self.tracked_class_ids = []

    def format_yolov5(self, input_frame): # put the image in square big enough
        row, col, _ = input_frame.shape
        _max = max(col, row)
        resized = np.zeros((_max, _max, 3), np.uint8)
        resized[0:row, 0:col] = input_frame
        # resize to 640x640, normalize to [0,1] and swap Red and Blue channels
        result = cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True)
        return result,resized
    
    def tlbr_to_tlwh(self, box):
        (x1, y1, x2, y2) = [int(coord) for coord in box]
        box = (x1, y1, x2 - x1, y2 - y1)
        return box

    def tlwh_to_tlbr(self, box):
        (x, y, w, h) = [int(coord) for coord in box]
        box = (x, y, x + w, y + h)
        return box
    
    def unwrap_detection(self, input_image, output_data):
        '''
        Outputs boxes in tlwh
        '''
        boxes = []
        confidences = []
        class_ids = []

        rows = output_data.shape[0]

        image_width, image_height, _ = input_image.shape

        x_factor = image_width / 640
        y_factor =  image_height / 640

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= self.dect_conf_thres: # ignore low detection confidence
                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > self.class_conf_thres): # ignore low class confidence
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        return boxes, confidences, class_ids

    def nms(self, in_boxes, in_confidences,in_class_ids):
        result_class_ids = []
        result_confidences = []
        result_boxes = []
        indexes = cv2.dnn.NMSBoxes(in_boxes, in_confidences, 0.25, 0.45) 
        for i in indexes:
            result_boxes.append(in_boxes[i])
            result_confidences.append(in_confidences[i])
            result_class_ids.append(in_class_ids[i])
        return result_boxes, result_confidences, result_class_ids
    
    def draw_bbox(self, current_frame, result_boxes, result_class_ids, tracking=False):
        class_list = []
        with open("/home/jeric/yolov5/classes.txt", "r") as f:
            class_list = [cname.strip() for cname in f.readlines()]

        colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

        for object_id, box in enumerate(result_boxes):
            # Unpack the bounding box coordinates
            (x, y, w, h) = [int(coord) for coord in box]
            class_id = result_class_ids[object_id]
            color = colors[class_id % len(colors)]
            #conf  = result_confidences[object_id]
            cv2.rectangle(current_frame, (x,y), (x+w,y+h), color, 2)
            cv2.rectangle(current_frame, (x,y-20), (x+w,y), color, -1)
            if tracking:
                cv2.putText(current_frame, f"{class_list[class_id]}: {str(object_id)}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
            else:
                cv2.putText(current_frame, class_list[class_id], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

    def detect(self, current_frame):
        print('DETECT: Receiving video frame')
        blob, input_image = self.format_yolov5(current_frame) #format image in to fit 640x640 and colour

        #set formatted image into net and pass through net
        self.net.setInput(blob)
        predictions = self.net.forward()
        output = predictions[0]

        boxes, confidences, class_ids = self.unwrap_detection(input_image,output) # unwrap detections

        result_boxes, result_confidences, result_class_ids = self.nms(boxes, confidences, class_ids) # NMS to remove dup and overlap
        
        return result_boxes, result_confidences, result_class_ids 

    def init_track(self, current_frame, result_boxes, result_confidences, result_class_ids):
        # create track for each bounding box
        for i in range(len(result_boxes)):
            tracker = cv2.legacy.TrackerKCF.create()
            self.trackers.add(tracker, current_frame, tuple(result_boxes[i]))
            self.tracked_class_ids.append(result_class_ids[i])

    def track(self, current_frame):
        print('TRACK: Receiving video frame')
        # Update the multi-object tracker
        success, boxes = self.trackers.update(current_frame)
        
        self.draw_bbox(current_frame, boxes, self.tracked_class_ids, tracking=True)

        # Write the frame into the file 'output.avi' 
        self.out.write(current_frame)
        
        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("camera", 1280, 720)
        cv2.imshow("camera", current_frame) # Display image
        cv2.waitKey(1)



def main(args=None):
    object_tracker = objectTracker()
    while 1:
        ret, current_frame = object_tracker.cap.read()

        if object_tracker.frame_count == 10:
            result_boxes, result_confidences, result_class_ids = object_tracker.detect(current_frame)
            object_tracker.init_track(current_frame,result_boxes, result_confidences, result_class_ids)
        else:
            # Start the timer
            start_time = cv2.getTickCount()
            object_tracker.track(current_frame)

            # Calculate the elapsed timE
            ticks = cv2.getTickCount() - start_time
            elapsed_time = (ticks / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
            print("Time taken for track:", elapsed_time, "milliseconds")
            
        object_tracker.frame_count += 1



if __name__ == '__main__':
    main()
