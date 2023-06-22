import scipy.io as sio
import numpy as np
import pandas as pd
import csv
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Convert .mat to .csv to multiple .txt')

# Add the input and output file arguments
parser.add_argument('input_mat_path', help='Input .mat file path')
parser.add_argument('output_folder_path', help='Output folder path')

# Parse the arguments
args = parser.parse_args()

input_mat_path = args.input_mat_path
video_name = input_mat_path.split('/')[-1].split('_ObjectGT.mat')[0]
output_folder_path = args.output_folder_path
output_csv = output_folder_path + video_name + ".csv"

data = sio.loadmat(input_mat_path)
print(data.keys())

# Get variable information from the .mat file
variables = sio.whosmat(input_mat_path)

print(variables)

# Access the structXML variable
struct_data = data['structXML']

# Print the dimensions and shape of the structXML ndarray
print('Dimensions:', struct_data.ndim)
print('Shape:', struct_data.shape)

# Extract the field names from the first row
field_names = struct_data[0, 0].dtype.names

print(f"fieldname {field_names}")

# Extract the data from the remaining rows
data_rows = struct_data[0, ].tolist()

# print(f"datarows {data_rows}")

# Determine the number of rows and columns
num_fields = len(field_names)


fw = 1920
fh = 1080
count = 0

for frame_num in range(struct_data.shape[1]):
    output_txt = output_folder_path + video_name + '_' + str(frame_num) + '.txt'
    with open(output_txt, 'w') as file:
        # Write the data
        for label in range(len(data_rows[frame_num][0])):
            # Extract type and bbox
            if data_rows[frame_num][1].size > 0:
                class_id = data_rows[frame_num][1][label][0]
                x = data_rows[frame_num][6][label][0]
                y = data_rows[frame_num][6][label][1]
                w = data_rows[frame_num][6][label][2]
                h = data_rows[frame_num][6][label][3]
                class_id -= 1 # Start from 0
                x = (x + w/2)/fw # normalise plus center
                y = (y + h/2)/fh # normalise plus center
                w = w/fw # normalise
                h = h/fh # normalise
                extracted_data = [str(class_id), str(x), str(y), str(w), str(h)]
            else:
                continue
            line = ' '.join(extracted_data)
            file.write(line + '\n')
    count += 1

print("Conversion to TXT complete.")
print(f"Total extracted: {count}")