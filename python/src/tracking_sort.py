'''
tracking script for python using SORT (kalman filter)

NOTE:
Does not support tracking without detection

Arguements are /path/to/tracking_ws and /path/to/video/file
You can change the model used and class list in the init of objectDetector class
You can change tracking settings by inputting arguments in Sort( max_age=1, min_hits=3, iou_threshold=0.3)
'''

from sort import *
from detection import *

def main(tracking_ws_path, input_video_path):

    #create instance of SORT
    mot_tracker = Sort() 
    object_detector = objectDetector(tracking_ws_path, input_video_path)
    rematch_rate = 1
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("camera", 1280, 720)
    cv2.cuda.setDevice(0) 

    while 1:
        # Start the timer
        start_time_cycle = cv2.getTickCount()

        sort_input_detections = np.empty((0,5))
        ret, current_frame = object_detector.cap.read()

        if not ret:
            print("Nothing Read")
            break

        # do detection and matching
        if object_detector.frame_count % rematch_rate == 0:
            start_time_detect = cv2.getTickCount()

            # pass through detector
            object_detector.detect(current_frame)

            ticks_detect = cv2.getTickCount() - start_time_detect
            elapsed_time_detect = (ticks_detect / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
            print("Time taken for detection:", elapsed_time_detect, "milliseconds")

            # # output in [[x1,y1,w,h], [....], ...] and [conf1, conf2, ...]
            # print(object_detector.detect_bboxes)
            # print(object_detector.detect_conf)

            # column stack to [[x1,y1,w,h,conf1],[..]] if detection is not empty
            if object_detector.detect_bboxes:
                sort_input_detections = np.column_stack((object_detector.detect_bboxes, object_detector.detect_conf))
                sort_input_detections[:, 2:4] += sort_input_detections[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        
            # print(sort_input_detections)

            start_time_track = cv2.getTickCount()

            # update SORT
            trackers = mot_tracker.update(sort_input_detections)

            ticks_track = cv2.getTickCount() - start_time_track
            elapsed_time_track = (ticks_track / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
            print("Time taken for track:", elapsed_time_track, "milliseconds")

        # # track without detection (DOES NOT WORK)
        # else:
        #     start_time_track = cv2.getTickCount()

        #     # update SORT
        #     trackers = mot_tracker.update()

        #     ticks_track = cv2.getTickCount() - start_time_track
        #     elapsed_time_track = (ticks_track / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
        #     print("Time taken for track:", elapsed_time_track, "milliseconds")

        # print(trackers)
        # for d in trackers:
        #   print('%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))

        # draw bbox
        colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

        for track in trackers:
            # Unpack the bounding box coordinates, convert from x1,y1,x2,y2 to tlwh
            (x, y, w, h) = int(track[0]),int(track[1]),int(track[2]-track[0]),int(track[3]-track[1])
            track_id = int(track[4])
            color = colors[track_id % len(colors)]
            #conf  = result_confidences[object_id]
            cv2.rectangle(current_frame, (x,y), (x+w,y+h), color, 2)
            cv2.rectangle(current_frame, (x,y-30), (x+w,y), color, -1)
            cv2.putText(current_frame, f"{track_id}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        
        # Write the frame into the file 'output.avi' 
        object_detector.out.write(current_frame)
        
        cv2.imshow("camera", current_frame) # Display image
        cv2.waitKey(1)
        
        object_detector.frame_count += 1

        # Calculate the elapsed time
        ticks_cycle = cv2.getTickCount() - start_time_cycle
        elapsed_time_cycle = (ticks_cycle / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
        print("Time taken:", elapsed_time_cycle, "milliseconds\n")
    
    object_detector.cap.release()
    object_detector.out.release()
    cv2.destroyAllWindows()

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
    tracking_ws_path = args.tracking_ws_path
    input_video_path = args.input_video_path

    main(tracking_ws_path, input_video_path)

    # Calculate the elapsed time
    total_ticks = cv2.getTickCount() - total_start_time
    total_elapsed_time = (total_ticks / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
    print("Total Time taken:", total_elapsed_time, "milliseconds\n")