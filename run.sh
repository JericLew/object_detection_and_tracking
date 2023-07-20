#!/bin/bash

# FOR Onshore frame extraction
# Source directory
tracking_ws_dir=~/tracking_ws/

# Source directory
video_input_dir=/media/jeric/1FC7-901F/dataset/VIS_Onboard/Videos/no_obj_label/

# Loop through the files in the source directory
for file_path in "$video_input_dir"/*
do
    # Check if the path corresponds to a file
    if [ -f "$file_path" ]; then
        # Extract the file name from the path
        file_name=$(basename "$file_path")
        
        # Run the Python command with the file path
        ./Detection "$tracking_ws_dir" "$file_path"
    fi
done

video_input_dir=/media/jeric/1FC7-901F/dataset/VIS_Onshore/Videos/no_obj_label/

# Loop through the files in the source directory
for file_path in "$video_input_dir"/*
do
    # Check if the path corresponds to a file
    if [ -f "$file_path" ]; then
        # Extract the file name from the path
        file_name=$(basename "$file_path")
        
        # Run the Python command with the file path
        ./Detection "$tracking_ws_dir" "$file_path"
    fi
done

# # Source directory
# video_input_dir=~/video_for_track/

# # Loop through the files in the source directory
# for file_path in "$video_input_dir"/*
# do
#     # Check if the path corresponds to a file
#     if [ -f "$file_path" ]; then
#         # Extract the file name from the path
#         file_name=$(basename "$file_path")
        
#         # Run the Python command with the file path
#         python ~/tracking_ws/python/src/tracking_sort.py "$tracking_ws_dir" "$file_path"
#     fi
# done


# # Source directory
# video_input_dir=~/video_for_track/

# # Loop through the files in the source directory
# for file_path in "$video_input_dir"/*
# do
#     # Check if the path corresponds to a file
#     if [ -f "$file_path" ]; then
#         # Extract the file name from the path
#         file_name=$(basename "$file_path")
        
#         # Run the Python command with the file path
#         ./TrackingAssociation "$tracking_ws_dir" "$file_path" CSRT
#     fi
# done

# # Source directory
# video_input_dir=/media/jeric/1FC7-901F/dataset/VIS_Onboard/Videos/

# # Loop through the files in the source directory
# for dir_path in "$video_input_dir"/*
# do
#     for file_path in "$dir_path"/*
#     do
#         # Check if the path corresponds to a file
#         if [ -f "$file_path" ]; then
#             # Extract the file name from the path
#             file_name=$(basename "$file_path")
            
#             # Run the Python command with the file path
#             ./TrackingAssociation "$tracking_ws_dir" "$file_path" MOSSE
#         fi
#     done
# done

# video_input_dir=/media/jeric/1FC7-901F/dataset/VIS_Onshore/Videos/

# for dir_path in "$video_input_dir"/*
# do
#     for file_path in "$dir_path"/*
#     do
#         # Check if the path corresponds to a file
#         if [ -f "$file_path" ]; then
#             # Extract the file name from the path
#             file_name=$(basename "$file_path")
            
#             # Run the Python command with the file path
#             ./TrackingAssociation "$tracking_ws_dir" "$file_path" MOSSE
#         fi
#     done
# done