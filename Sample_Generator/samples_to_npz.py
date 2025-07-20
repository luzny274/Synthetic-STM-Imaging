
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

import png

import os
import os.path
import re

import gc

dataset_folder = "WSe2_dataset/"

def find_highest_number_in_filenames(folder_path):
    images_by_number = {}

    highest_number = 0
    number_pattern = re.compile(r'\d+')

    try:
        # List all files in the given folder
        filenames = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Iterate through each filename in the folder
    for filename in filenames:
        # Find all numbers in the filename

        filename_slashes = [pos for pos, char in enumerate(filename) if char == '\\' or char == '/']
        if len(filename_slashes) > 0:
            filename = filename[filename_slashes[-1]:]
        
        numbers_in_filename = number_pattern.findall(filename)
        
        # Convert found numbers to integers and find the maximum number
        if numbers_in_filename:
            numbers = [int(num) for num in numbers_in_filename]
            max_number_in_filename = max(numbers)

            if ".png" in filename:
                if max_number_in_filename not in images_by_number:
                    images_by_number[max_number_in_filename] = []
                images_by_number[max_number_in_filename].append(filename)
            
            # Update the highest number found so far
            if max_number_in_filename > highest_number:
                highest_number = max_number_in_filename

    return highest_number, images_by_number

##Get a sample
def get_sample(ind, images):
    if os.path.isfile(dataset_folder + str(ind) + "_params.npz"):
        npz_file = np.load(dataset_folder + str(ind) + "_params.npz")

        defect_count  = npz_file["defect_count"]
        vacancy_count = npz_file["vacancy_count"]
        dopant_count  = npz_file["dopant_count"]
        vacancy_pos   = npz_file["vacancy_pos"]
        dopant_pos    = npz_file["dopant_pos"]
        K_P           = npz_file["K_P"]
        K_D           = npz_file["K_D"]
        sample_size   = npz_file["sample_size"]

        def load_png(filename):
            reader = png.Reader(dataset_folder + filename)
            pngdata = reader.read()
            px_array = np.array(list(pngdata[2])) / 255
            return(px_array)

        img_arr      = [load_png(filename) for filename in images if "voltage_profile.png" in filename]
        ground_truth = [load_png(filename) for filename in images if "ground_truth.png" in filename]

        if len(img_arr) != 1 or len(ground_truth) != 1:
            print("ERROR::incorrect amount of images associated with sample", ind)
        img_arr = img_arr[0]
        ground_truth = ground_truth[0]

        X  = []
        Y1 = []
        Y2 = []
        Y3 = []
        S  = []

        res = ground_truth.shape[0]
        dopant_pos /= res
        vacancy_pos /= res

        X.append(img_arr.reshape((1, res, res)))
        Y1.append(np.array([defect_count]))
        Y2.append(ground_truth.reshape((1, res, res, 3)))
        Y3.append(np.array([K_P]))
            
        defects = []
        if dopant_pos.shape[0]  > 0: defects.append(dopant_pos )
        if vacancy_pos.shape[0] > 0: defects.append(vacancy_pos)
        if len(defects) > 1        : defects = np.concatenate(defects, axis = 0)
        S.append(defects)

        return [X, Y1, Y2, Y3, S]
    else:
        return [None, None, None, None]

#Load dataset
ind = 0
X, Y1, Y2, Y3, S = [], [], [], [], []
max_ind, images_by_number = find_highest_number_in_filenames(dataset_folder)
for ind in range(0, max_ind):
    print("Loaded {}/{} samples".format(ind, max_ind), end="\r")
    if ind in images_by_number:
        x, y1, y2, y3, s = get_sample(ind, images_by_number[ind])
        if x is not None:
            X  += x
            Y1 += y1
            Y2 += y2
            Y3 += y3
            S  += s
    else:
        print("Skipping non-existent sample number", ind)
print("\n")
X  = np.concatenate(X, axis = 0)
Y1 = np.array(Y1)
Y2 = np.concatenate(Y2)
Y3 = np.array(Y3)

S_np = np.empty((len(S)), dtype=object)
for i in range(len(S)):
    S_np[i] = S[i]

print(X.shape, Y1.shape, Y2.shape, Y3.shape, S_np.shape)
np.savez("WSe2_dataset.npz", X=X, Y1=Y1, Y2=Y2, Y3=Y3, S=S_np)