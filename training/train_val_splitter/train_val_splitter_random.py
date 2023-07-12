import os
import random
import shutil

# Specify the paths to the image folder, label folder, train folder, and validation folder
image_source_folder = '/home/jeric/datasets/merge_class_copy/images/'
label_source_folder = '/home/jeric/datasets/merge_class_copy/labels/'
image_train_folder = '/home/jeric/datasets/merge_class_random_split/images/train/'
image_val_folder = '/home/jeric/datasets/merge_class_random_split/images/val/'
label_train_folder = '/home/jeric/datasets/merge_class_random_split/labels/train/'
label_val_folder = '/home/jeric/datasets/merge_class_random_split/labels/val/'
split_ratio = 0.7  # 70% for training, 30% for validation

def split_data(image_source_folder, image_train_folder, image_val_folder, split_ratio):
    # Create the train and validation folders if they don't exist
    os.makedirs(image_train_folder, exist_ok=True)
    os.makedirs(image_val_folder, exist_ok=True)
    
    # Get the list of image files in the source folder
    image_files = [file for file in os.listdir(image_source_folder) if file.endswith('.jpg')]
    
    # Shuffle the image files randomly
    random.shuffle(image_files)
    
    # Calculate the split index
    split_index = int(len(image_files) * split_ratio)
    
    # Copy files to the train folder
    for file in image_files[:split_index]:
        src_path = os.path.join(image_source_folder, file)
        dst_path = os.path.join(image_train_folder, file)
        shutil.copyfile(src_path, dst_path)
    
    # Copy files to the validation folder
    for file in image_files[split_index:]:
        src_path = os.path.join(image_source_folder, file)
        dst_path = os.path.join(image_val_folder, file)
        shutil.copyfile(src_path, dst_path)

def sort_labels(image_source_folder, label_source_folder, image_train_folder, image_val_folder, label_train_folder, label_val_folder):
    # Create the train and validation folders if they don't exist
    os.makedirs(label_train_folder, exist_ok=True)
    os.makedirs(label_val_folder, exist_ok=True)

    # set method

    # Get the list of image files in the image folder
    image_files = set(os.listdir(image_source_folder))
    
    # Get the list of label files in the label folder
    label_files = os.listdir(label_source_folder)

    # Create sets for faster lookup
    train_files = set(os.listdir(image_train_folder))
    val_files = set(os.listdir(image_val_folder))

    for label_file in label_files:
        # Extract the image file name from the label file
        image_file = label_file.replace('.txt', '.jpg')

        # Determine the destination folder based on the presence of the image file in train or val folders
        if image_file in train_files:
            dst_folder = label_train_folder
        elif image_file in val_files:
            dst_folder = label_val_folder
        else:
            continue  # Skip if the image file is not found in either train or val folder

        # Copy the label file to the destination folder
        src_label_file_path = os.path.join(label_source_folder, label_file)
        dst_label_file_path = os.path.join(dst_folder, label_file)
        shutil.copyfile(src_label_file_path, dst_label_file_path)

'''
    # dict method

    # Get the list of image files in the image folder
    image_files = [image_file for image_file in os.listdir(image_source_folder) if image_file.endswith('.jpg')]

    count = 0
    print("Adding to dictionary")
    file_map = {}
    for image_file in image_files:
        print(f"adding to dict {count}")
        count += 1
        if image_file in os.listdir(image_train_folder):
            file_map[image_file] = label_train_folder
        elif image_file in os.listdir(image_val_folder):
            file_map[image_file] = label_val_folder
    print("Finished dictionary")
    
    # Copy the label files to the destination folders
    for label_file in os.listdir(label_source_folder):
        if label_file.endswith('.txt'):
            image_file = label_file.replace('.txt', '.jpg')
            if image_file in file_map:
                src_path = os.path.join(label_source_folder, label_file)
                dst_folder = file_map[image_file]
                dst_path = os.path.join(dst_folder, label_file)
                shutil.copyfile(src_path, dst_path)
'''
''' 
    # old method

    # Get the list of image files in the image folder
    image_files = [image_file for image_file in os.listdir(image_source_folder) if image_file.endswith('.jpg')]

    for image_file in image_files:
        # Generate the corresponding label file path
        label_file = image_file.replace('.jpg', '.txt')
        label_file_path = os.path.join(label_source_folder, label_file)

        # Determine the destination folder based on the image file location
        if image_file in os.listdir(image_train_folder):
            dst_folder = label_train_folder
        elif image_file in os.listdir(image_val_folder):
            dst_folder = label_val_folder
        else:
            continue  # Skip if the image file is not found in either train or val folder

        # Copy the label file to the destination folder
        dst_label_file_path = os.path.join(dst_folder, label_file)
        shutil.copyfile(label_file_path, dst_label_file_path)
'''

print("Starting Image Random Split")
# Call the split_data function
split_data(image_source_folder, image_train_folder, image_val_folder, split_ratio)
print("Image Random Split Done")
print("Starting Label Sorting")
# Call the sort_labels function
sort_labels(image_source_folder, label_source_folder, image_train_folder, image_val_folder, label_train_folder, label_val_folder)
print("Label Sorting Done")