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

frame_count = 0
input_mat_path = args.input_mat_path
video_name = input_mat_path.split('/')[-1].split('_ObjectGT.mat')[0]
output_folder_path = args.output_folder_path
output_csv = output_folder_path + video_name + ".csv"
output_txt = output_folder_path + video_name + '_' + str(frame_count) + '.txt'

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

def write_csv():
    # Open the output CSV file for writing
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row with the field names
        writer.writerow(field_names)

        # Write the data 
        for i in range(struct_data.shape[1]):

            # # to extract type and bbox 
            # if data_rows[i][1].size > 0:
            #     extracted_data = [data_rows[i][1][0][0], data_rows[i][6][0][0], data_rows[i][6][0][1] ,data_rows[i][6][0][2] ,data_rows[i][6][0][3]]
            # else:
            #     extracted_data = [0, 0, 0, 0, 0]
            # writer.writerow(extracted_data) 

            # to extract all data
            writer.writerow(data_rows[i][j] for j in range(len(field_names))) 
    print(f"Conversion to csv complete.")

write_csv()