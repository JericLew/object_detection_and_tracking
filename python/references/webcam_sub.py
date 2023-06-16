# Basic ROS 2 program to subscribe to real-time streaming 
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
  
# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import numpy as np
 
class ImageSubscriber(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_subscriber')
      
    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.
    self.subscription = self.create_subscription(
      Image, 
      'video_frames', 
      self.listener_callback, 10)
    self.subscription # prevent unused variable warning
      
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()

    # test yolov5 object detection stuff
    self.net = cv2.dnn.readNet('/home/jeric/yolov5/yolov5s.onnx')

    
    self.count = 0



# test yolov5 object detection stuff
  def format_yolov5(self, source):
    # put the image in square big enough
    row, col, _ = source.shape
    _max = max(col, row)
    resized = np.zeros((_max, _max, 3), np.uint8)
    resized[0:row, 0:col] = source
    
    # resize to 640x640, normalize to [0,1] and swap Red and Blue channels
    result = cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True)
    
    return result,resized

  def unwrap_detection(self, input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4: #remove those below 0.4 confidence

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    return class_ids, confidences, boxes
  
  def listener_callback(self, data):
    """
    Callback function. 
    """
    # Display the message on the console
    self.get_logger().info('Receiving video frame')
 
    # Convert ROS Image message to OpenCV image
    current_frame = self.br.imgmsg_to_cv2(data)
    
    # only do detection every _ frames (NEED A WAY TO PRESERVE BB)
    if self.count%1 == 0:
      #format image in to fit 640x640 and colour
      blob, input_image = self.format_yolov5(current_frame)
      
      #set formatted image into net and pass through net
      self.net.setInput(blob)
      predictions = self.net.forward()
      output = predictions[0]

      #unwrap detections
      class_ids, confidences, boxes = self.unwrap_detection(input_image,output)
    
      #NMS to remove dup and overlap
      indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

      result_class_ids = []
      result_confidences = []
      result_boxes = []
      for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

      #printing result
      class_list = []
      with open("/home/jeric/yolov5/classes.txt", "r") as f:
          class_list = [cname.strip() for cname in f.readlines()]

      colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

      for i in range(len(result_class_ids)):

          box = result_boxes[i]
          class_id = result_class_ids[i]

          color = colors[class_id % len(colors)]

          conf  = result_confidences[i]

          cv2.rectangle(current_frame, box, color, 2)
          cv2.rectangle(current_frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
          cv2.putText(current_frame, class_list[class_id], (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
    self.count += 1
    # Display image
    cv2.imshow("camera", current_frame)
    
    cv2.waitKey(1)

  

  
def main(args=None):

  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  image_subscriber = ImageSubscriber()
  
  # Spin the node so the callback function is called.
  rclpy.spin(image_subscriber)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_subscriber.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
