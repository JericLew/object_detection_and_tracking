import os
import re
import numpy as np
import cv2

def draw_bbox(current_frame, result_boxes, result_class_ids, tracking=False):
    class_list = []
    with open("/home/jeric/tracking_ws/classes/classes_semi_merge.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]

    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    if tracking==False:
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
    else:
        for object_id, box in enumerate(result_boxes):
            if box == None:
                continue
            # Unpack the bounding box coordinates
            (x, y, w, h) = [int(coord) for coord in box]
            class_id = result_class_ids[object_id]
            color = colors[class_id % len(colors)]
            #conf  = result_confidences[object_id]
            cv2.rectangle(current_frame, (x,y), (x+w,y+h), color, 2)
            cv2.rectangle(current_frame, (x,y-30), (x+w,y), color, -1)
            cv2.putText(current_frame, class_list[class_id], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)


# video writer
fw = 1920
fh = 1080
out = cv2.VideoWriter(f"/home/jeric/train_labels.mp4",cv2.VideoWriter_fourcc('m','p','4','v'),30,(fw,fh)) # create writer obj


# Directory containing the image files
image_dir = "/home/jeric/datasets/semi_merge/images/"
label_dir = "/home/jeric/datasets/semi_merge/labels/"

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
    
    # Set the frame rate
    frame_rate = 300
    
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
        # cv2.imshow("Video", image)

        # Save the video
        out.write(image)

        # # Wait for the specified time between frames
        # delay = int(1000 / frame_rate)
        # if cv2.waitKey(delay) == ord('q'):
        #     break
        cv2.waitKey(1)
    
    # Clean up
    cv2.destroyAllWindows()

    
# # Sort the files in the directory
# image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")],
#                      key=lambda x: int(re.findall(r'\d+(?=\.)', x)[0]))

# # Set the frame rate
# frame_rate = 1

# # Iterate over the image files
# for image_file in image_files:
#     print(image_file)
#     result_boxes = []
#     result_class_ids = []
#     # Load the image
#     image_path = os.path.join(image_dir, image_file)
#     image = cv2.imread(image_path)
    
#     label_file = image_file.replace(".jpg", ".txt")
#     label_path = os.path.join(label_dir, label_file)

#     if os.path.isfile(label_path):
#         with open(label_path, 'r') as f:
#             for line in f:
#                 # Extract values from each line as needed
#                 values = line.strip().split(',')
#                 # Process the values here
#                 # result_class_ids.append(values[0])
#                 # result_boxes.append()
#                 # Add values to the label matrix
#                 result_boxes.append(values)
    
    
                

#     # Display the image
#     cv2.imshow("Video", image)
    
#     # Wait for the specified time between frames
#     delay = int(1000 / frame_rate)
#     if cv2.waitKey(delay) == ord('q'):
#         break

# # Clean up
# cv2.destroyAllWindows()