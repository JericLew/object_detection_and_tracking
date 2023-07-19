import cv2 
import argparse
import os
start = cv2.getTickCount()

# Create the argument parser
parser = argparse.ArgumentParser(description='Extract frames from video')

# Add the input and output file arguments
parser.add_argument('input_video_path', help='/path/to/video/file/video.xxx')
parser.add_argument('output_folder_path', help='/path/to/output/folder/')

# Parse the arguments
args = parser.parse_args()

input_video_path = args.input_video_path
video_name = input_video_path.split('/')[-1]
video_name_no_ext = video_name.split('.')[0]
output_folder_path = args.output_folder_path

# make dir if it doesnt exist
os.makedirs(output_folder_path, exist_ok=True)

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video Name: {video_name}")
print(f"FPS: {fps}")
print(f"Width x Height: {width} x {height}")

frameCount = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_name = output_folder_path + video_name_no_ext + "_" + str(frameCount) + ".jpg"
    cv2.imwrite(frame_name, frame)
    
    frameCount += 1

cap.release()

print("Total frames extracted:", frameCount)

end = cv2.getTickCount()
elapsedTime = (end - start) / cv2.getTickFrequency()
print("Total Elapsed Time:", elapsedTime, "s")
