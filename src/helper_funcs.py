import cv2 
import numpy as np
from scipy.optimize import linear_sum_assignment

def format_yolov5(input_frame): # put the image in square big enough
    row, col, _ = input_frame.shape
    _max = max(col, row)
    resized = np.zeros((_max, _max, 3), np.uint8)
    resized[0:row, 0:col] = input_frame
    # resize to 640x640, normalize to [0,1] and swap Red and Blue channels
    result = cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True)
    return result,resized

def tlbr_to_tlwh(box):
    (x1, y1, x2, y2) = [int(coord) for coord in box]
    box = (x1, y1, x2 - x1, y2 - y1)
    return box

def tlwh_to_tlbr(box):
    (x, y, w, h) = [int(coord) for coord in box]
    box = (x, y, x + w, y + h)
    return box


def unwrap_detection_numpy(input_image, output_data, detect_conf_thres, class_conf_thres):
    boxes = []
    confidences = []
    class_ids = []

    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor = image_height / 640

    valid_indices = np.where(output_data[:, 4] >= detect_conf_thres)[0]
    valid_classes = np.argmax(output_data[valid_indices, 5:], axis=1)
    valid_scores = output_data[valid_indices, 5:][np.arange(len(valid_indices)), valid_classes]
    valid_mask = valid_scores > class_conf_thres

    valid_indices = valid_indices[valid_mask]
    valid_classes = valid_classes[valid_mask]
    valid_scores = valid_scores[valid_mask]

    valid_rows = output_data[valid_indices]
    valid_boxes = valid_rows[:, :4]

    valid_boxes[:, 0] = (valid_boxes[:, 0] - 0.5 * valid_boxes[:, 2])
    valid_boxes[:, 1] = (valid_boxes[:, 1] - 0.5 * valid_boxes[:, 3])
    valid_boxes = valid_boxes.astype(int)
    valid_boxes[:, 0] = valid_boxes[:, 0] * x_factor
    valid_boxes[:, 1] = valid_boxes[:, 1] * y_factor

    valid_boxes[:, 2] = valid_boxes[:, 2] * x_factor
    valid_boxes[:, 3] = valid_boxes[:, 3] * y_factor

    boxes = valid_boxes.tolist()
    confidences = valid_scores.tolist()
    class_ids = valid_classes.tolist()

    return boxes, confidences, class_ids

def unwrap_detection(input_image, output_data, detect_conf_thres, class_conf_thres):
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
        if confidence >= detect_conf_thres: # ignore low detection confidence
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > class_conf_thres): # ignore low class confidence
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

def nms(in_boxes, in_confidences,in_class_ids, detect_conf_thres, nms_thres):
    result_class_ids = []
    result_confidences = []
    result_boxes = []
    indexes = cv2.dnn.NMSBoxes(in_boxes, in_confidences, detect_conf_thres, nms_thres) 
    for i in indexes:
        result_boxes.append(in_boxes[i])
        result_confidences.append(in_confidences[i])
        result_class_ids.append(in_class_ids[i])
    return result_boxes, result_confidences, result_class_ids

def draw_bbox(current_frame, result_boxes, result_class_ids, tracking=False):
    class_list = []
    with open("/home/jeric/tracking_ws/classes/classes.txt", "r") as f:
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
            cv2.rectangle(current_frame, (x,y-20), (x+w,y), color, -1)
            if tracking:
                cv2.putText(current_frame, f"{class_list[class_id]}: {str(object_id)}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
            else:
                cv2.putText(current_frame, class_list[class_id], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

        
def calculate_iou(bb1, bb2):
    '''
    input in tlwh
    '''
    bb1 = tlwh_to_tlbr(bb1)
    bb2 = tlwh_to_tlbr(bb2)
    x1, y1, x2, y2 = bb1
    x3, y3, x4, y4 = bb2
    xa = max(x1, x3)
    ya = max(y1, y3)
    xb = min(x2, x4)
    yb = min(y2, y4)
    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    area_bb1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area_bb2 = (x4 - x3 + 1) * (y4 - y3 + 1)
    iou = intersection / float(area_bb1 + area_bb2 - intersection)
    return iou


def hung_algo(track_boxes, detect_boxes):
    iou_threshold = 0.7
    # Create an empty list to store matches
    matches = []
    unmatched_tracks = list(range(len(track_boxes)))
    unmatched_detections = list(range(len(detect_boxes)))

    # Create a cost matrix for matching using IOU (Intersection over Union)
    cost_matrix = np.zeros((len(track_boxes), len(detect_boxes)))

    # Calculate IOU for each track-detection pair and populate the cost matrix
    for i, track in enumerate(track_boxes):
        for j, detection in enumerate(detect_boxes):
            if track == None:
                cost_matrix[i, j] = 99
            else:
                iou = calculate_iou(track, detection)
                # Assign IOU to the cost matrix
                cost_matrix[i, j] = 1 - iou

    # Perform matching using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create matches based on the matching results
    for row, col in zip(row_ind, col_ind):
        if cost_matrix[row, col] < iou_threshold:  # Set a threshold for accepting matches
            matches.append((row, col))
    
    for track_id, detect_id in matches:
        unmatched_tracks.remove(track_id)       
        unmatched_detections.remove(detect_id)

    return matches, unmatched_tracks, unmatched_detections # matches in (track_id, detect_id)

