import scipy.io as sio
import numpy as np
import pandas as pd

import os
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Merge classes by going through all txt in a directory and editing them')

# Add the input and output file arguments
parser.add_argument('directory', help='/path/to/directory')
# Parse the arguments
args = parser.parse_args()
directory = args.directory

file_count = 0

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
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