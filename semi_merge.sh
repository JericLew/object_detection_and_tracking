#!/bin/bash
''' 
This bash script extracts frames and labels for training.

This is for semi_merge, which only merges the swimmer,aerial and other class into background

Run this script when in the directory ~/tracking_ws/

Make sure that
 - ~/datasets/ is downloaded
 - "semi_merge" folder which has "images" and "labels" is created

Run train_val_splitter_random.py after generation of files
'''
# FOR Onshore frame extraction
# Source directory
source_dir=~/datasets/VIS_Onshore/Videos/train

# Destination directory
destination_dir=~/datasets/semi_merge/images/

# Loop through the files in the source directory
for file_path in "$source_dir"/*
do
    # Check if the path corresponds to a file
    if [ -f "$file_path" ]; then
        # Extract the file name from the path
        file_name=$(basename "$file_path")
        
        # Run the Python command with the file path
        python ./training/frame_extractor/frame_extractor.py "$file_path" "$destination_dir"
    fi
done

# FOR Onboard frame extraction
# Source directory
source_dir=~/datasets/VIS_Onboard/Videos/train

# Destination directory
destination_dir=~/datasets/semi_merge/images/

# Loop through the files in the source directory
for file_path in "$source_dir"/*
do
    # Check if the path corresponds to a file
    if [ -f "$file_path" ]; then
        # Extract the file name from the path
        file_name=$(basename "$file_path")
        
        # Run the Python command with the file path
        python ./training/frame_extractor/frame_extractor.py "$file_path" "$destination_dir"
    fi
done

# FOR Onshore label extraction
# Source directory
source_dir=~/datasets/VIS_Onshore/ObjectGT/train

# Destination directory
destination_dir=~/datasets/semi_merge/labels/

# Loop through the files in the source directory
for file_path in "$source_dir"/*
do
    # Check if the path corresponds to a file
    if [ -f "$file_path" ]; then
        # Extract the file name from the path
        file_name=$(basename "$file_path")
        
        # Run the Python command with the file path
        python ./training/matlab_convertor/matlab_to_yolo_labels.py "$file_path" "$destination_dir"
    fi
done

# FOR Onboard label extraction
# Source directory
source_dir=~/datasets/VIS_Onboard/ObjectGT/train

# Destination directory
destination_dir=~/datasets/semi_merge/labels/

# Loop through the files in the source directory
for file_path in "$source_dir"/*
do
    # Check if the path corresponds to a file
    if [ -f "$file_path" ]; then
        # Extract the file name from the path
        file_name=$(basename "$file_path")
        
        # Run the Python command with the file path
        python ./training/matlab_convertor/matlab_to_yolo_labels.py "$file_path" "$destination_dir"
    fi
done

# FOR label merge
python ./training/merge_class/semi_merge.py ~/datasets/semi_merge/labels/
