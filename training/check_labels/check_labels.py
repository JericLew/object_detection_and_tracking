'''
This python script is used to look through all the frames and labels of a dataset
and output a video of all the frames with Bounding Box Labels

Please change the below paths and input video details as required
'''

import os
import re
import numpy as np
import cv2
import argparse

# Paths for files and folders
image_dir = "/home/jeric/datasets/semi_merge/images/"
label_dir = "/home/jeric/datasets/semi_merge/labels/"
path_to_output_folder = "/home/jeric/train_labels.mp4"
path_to_class_list = "/home/jeric/tracking_ws/classes/classes_semi_merge.txt"
show_video = True

# video writer
fw = 1920
fh = 1080
frame_rate = 30
out = cv2.VideoWriter(path_to_output_folder,cv2.VideoWriter_fourcc('m','p','4','v'),30,(fw,fh)) # create writer obj

def draw_bbox(current_frame, result_boxes, result_class_ids, tracking=False):
    class_list = []
    with open(path_to_class_list, "r") as f:
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
        cv2.putText(current_frame, class_list[class_id], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

# Get the image files and sort them numerically within each video group
video_groups = {}
for image_file in os.listdir(image_dir):
    if image_file.endswith(".jpg"):
        video_name = re.findall(r'MVI_\d+', image_file)[0]
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(image_file)

for video_name, image_files in video_groups.items():
    # sort img files in video group according to the numbers at the back before the . extension
    sorted_image_files = sorted(image_files, key=lambda x: int(re.findall(r'\d+(?=\.)', x)[0]))

    # Iterate over the sorted image files within each video group
    for image_file in sorted_image_files:
        # Load the image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        label_file = image_file.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label_file)

        result_boxes = []
        result_class_ids = []   

        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # Extract values from each line as needed
                    values = line.strip().split(' ')
                    # print(values)
                    # Process the values here
                    result_class_ids.append(int(values[0]))
                    # result_boxes.append()
                    # Add values to the label matrix
                    x, y, w, h = float(values[1]), float(values[2]), float(values[3]), float(values[4])
                    left = int((x - 0.5 * w) * fw)
                    top = int((y - 0.5 * h) * fh)
                    width = int(w * fw)
                    height = int(h * fh)
                    box = np.array([left, top, width, height])
                    result_boxes.append(box)

        draw_bbox(image, result_boxes, result_class_ids)

        # Display the image
        if show_video:
            cv2.imshow("Video", image)
            # Wait for the specified time between frames
            delay = int(1000 / frame_rate)
            if cv2.waitKey(delay) == ord('q'):
                break
        
        # Save the video
        out.write(image)

# Clean up
cv2.destroyAllWindows()