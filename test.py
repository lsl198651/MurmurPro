import os

from util.utils_dataset import csv_to_dict

if __name__ == "__main__":
    output_path = r"D:\Shilong\new_murmur\02_dataset\01_s1s2_4k\npyFile_padded\organized_data"
    data_path = r"D:\Shilong\new_murmur\02_dataset\01_s1s2_4k\npyFile_padded\index_files01_norm"
    for root, dir, file in os.walk(output_path):
        for file in file:
            # if file.endswith(".csv"):
            # print('processing file:', file)
            # get_id_position_org(root, output_path, file)
            print('processing file:', file)
            a = csv_to_dict(root, file)
