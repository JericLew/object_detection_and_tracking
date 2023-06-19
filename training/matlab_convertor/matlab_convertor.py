import scipy.io as sio
import numpy as np
import pandas as pd
import csv

data = sio.loadmat('/home/jeric/VIS_Onboard/ObjectGT/MVI_0790_VIS_OB_ObjectGT.mat')
print(data.keys())

# Get variable information from the .mat file
variables = sio.whosmat('/home/jeric/VIS_Onboard/ObjectGT/MVI_0790_VIS_OB_ObjectGT.mat')

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
    output_file = 'output.csv'
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row with the field names
        writer.writerow(field_names)

        # Write the data 
        for i in range(struct_data.shape[1]):

            # to extract type and bbox 
            if data_rows[i][1][0]:
                extracted_data = [data_rows[i][1][0][0], data_rows[i][6][0][0], data_rows[i][6][0][1] ,data_rows[i][6][0][2] ,data_rows[i][6][0][3]]
            else:
                extracted_data = [0, 0, 0, 0, 0]
            writer.writerow(extracted_data) 

            
            # # to extract all data
            # writer.writerow(data_rows[i][j] for j in range(len(field_names))) 
    print(f"Conversion complete. Saved as {output_file}")

def convert_txt():
    input_csv = 'output.csv'
    output_txt = 'output.txt'

    with open(input_csv, 'r') as file:
        csv_reader = csv.reader(file)
        rows = list(csv_reader)

    with open(output_txt, 'w') as file:
        for row in rows:
            line = ' '.join(row)
            file.write(line + '\n')

    print(f"Conversion complete. Saved as {output_txt}")

write_csv()
convert_txt()