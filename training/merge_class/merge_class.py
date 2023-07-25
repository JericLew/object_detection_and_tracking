'''
This python script is used to go through all .txt yolo label files in a directory
and update the classes of different objects to merge or remove them.

This script merges the 10 classes in the Singapore Maritime Dataset to 3 classes.

Vessesl:       Vessel, Ferry
Small Boats:   Speedboat. Boat, Kayak, Sailboat
Buoy:          Buoy
Background:    Swimmer, Aerial, Others

Input is the path to to folder containing labels

See merge_class.sh or semi_merge.sh to see how to use a bash script to automate
the process of preparing the dataset for training
'''

import scipy.io as sio
import numpy as np
import pandas as pd

import os
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Merge classes by going through all txt in a path_to_labels and editing them')

# Add the input and output file arguments
parser.add_argument('path_to_labels', help='/path/to/labels/')
# Parse the arguments
args = parser.parse_args()
path_to_labels = args.path_to_labels

file_count = 0

# Iterate over all files in the path_to_labels
for filename in os.listdir(path_to_labels):
    if filename.endswith('.txt'):
        filepath = os.path.join(path_to_labels, filename)
        with open(filepath, 'r') as file:
            lines = file.readlines()

        modified_lines = []

        for line in lines:
            elements = line.split()
            class_id = int(elements[0])  # Convert the first element to an integer

            # Modify the first element based on its value
            if class_id == 0:
                modified_class_id = 0
            elif class_id == 1:
                modified_class_id = 2
            elif class_id == 2:
                modified_class_id = 0
            elif class_id == 3:
                modified_class_id = 1
            elif class_id == 4:
                modified_class_id = 1
            elif class_id == 5:
                modified_class_id = 1
            elif class_id == 6:
                modified_class_id = 1
            elif class_id == 7:
                continue
            elif class_id == 8:
                continue
            elif class_id == 9:
                continue

            elements[0] = str(modified_class_id)  # Convert the modified first element back to a string
            modified_line = ' '.join(elements)
            modified_lines.append(modified_line)

        with open(filepath, 'w') as file:
            file.write('\n'.join(modified_lines))
            file_count += 1

    
print("Class merge completed")
print(f"Total .txt gone through: {file_count}")
