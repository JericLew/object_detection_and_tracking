from sort import *
from detection import *

#create instance of SORT
mot_tracker = Sort() 

if __name__ == '__main__':
    object_detector = objectDetector()
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("camera", 1280, 720)

    while 1:
        # Start the timer
        start_time = cv2.getTickCount()

        # format should be  [[x1,y1,x2,y2,conf],[..]]
        sort_input_detections = np.empty((0,5))
        ret, current_frame = object_detector.cap.read()
            
        if not ret:
            break 

        # pass through detector
        object_detector.detect(current_frame)

        # output in [[x1,y1,w,h], [....], ...] and [...]
        # print(object_detector.detect_bboxes)
        # print(object_detector.detect_conf)

        # column stack to [[x1,y1,w,h,conf],[..]]
        if object_detector.detect_bboxes:
            sort_input_detections = np.column_stack((object_detector.detect_bboxes, object_detector.detect_conf))
            sort_input_detections[:, 2:4] += sort_input_detections[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]

        # print(sort_input_detections)
        start_time_track = cv2.getTickCount()
        # update SORT
        trackers = mot_tracker.update(sort_input_detections)
        ticks_track = cv2.getTickCount() - start_time_track
        elapsed_time_track = (ticks_track / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
        print("Time taken for track:", elapsed_time_track, "milliseconds\n")

        # print(trackers)
        # for d in trackers:
        #   print('%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))

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
        
        # Calculate the elapsed time
        ticks = cv2.getTickCount() - start_time
        elapsed_time = (ticks / cv2.getTickFrequency()) * 1000  # Convert to milliseconds
        print("Time taken for Detection:", elapsed_time, "milliseconds\n")
    
    object_detector.cap.release()
    object_detector.out.release()
    cv2.destroyAllWindows()
