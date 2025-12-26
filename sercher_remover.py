import os
import pandas as pd

def collect_yaml_files_by_folder(root_dir):
    yaml_files_by_folder = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        yaml_files = [f.split(".")[0] for f in filenames if f.endswith('.yaml')]
        # Remove .png files from the directory
        for f in filenames:
            if f.endswith('.png'):
                os.remove(os.path.join(dirpath, f))
        if yaml_files:
            new_dir_path = dirpath.split("\\")[-1]
                                         
            yaml_files_by_folder[new_dir_path] = yaml_files

    return yaml_files_by_folder

# Example usage:
root_directory = r"C:\Users\mohammed.amirifard\Desktop\Github_Projects\Cooperative_Perception\testing_data - Copy"
yaml_files = collect_yaml_files_by_folder(root_directory)

import csv

def save_dict_columnwise(data, output_file):
    # Get the maximum number of files in any folder
    max_files = max(len(files) for files in data.values())

    # Create header: 'Folder', 'File_1', 'File_2', ...
    header = ['Folder'] + [f'File_{i+1}' for i in range(max_files)]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for folder, files in data.items():
            row = [folder] + files + [''] * (max_files - len(files))  # pad with empty strings
            writer.writerow(row)

# Example usage:
save_dict_columnwise(yaml_files, 'yaml_files_columnwise.csv')
