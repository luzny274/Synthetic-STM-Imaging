import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import matplotlib.pyplot as plt
import copy
import time

import png

import os
import os.path
import re

import gc

import pytorch_utils

from scipy.optimize import linear_sum_assignment

def find_highest_number_in_filenames(folder_path):
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
        numbers_in_filename = number_pattern.findall(filename)
        
        # Convert found numbers to integers and find the maximum number
        if numbers_in_filename:
            numbers = [int(num) for num in numbers_in_filename]
            max_number_in_filename = max(numbers)
            
            # Update the highest number found so far
            if max_number_in_filename > highest_number:
                highest_number = max_number_in_filename

    return highest_number

##Get a sample
def get_sample(ind):
    if os.path.isfile("WSe2_samples/" + str(ind) + "_params.npz"):
        npz_file = np.load("WSe2_samples/" + str(ind) + "_params.npz")
        defect_count        = npz_file["defect_count"]
        vacancy_count       = npz_file["vacancy_count"]
        dopant_count        = npz_file["dopant_count"]
        vacancy_pos         = npz_file["vacancy_pos"]
        dopant_pos          = npz_file["dopant_pos"]
        vacancy_site        = npz_file["vacancy_site"]
        dopant_site         = npz_file["dopant_site"]
        site_vacancy_counts = npz_file["site_vacancy_counts"]
        site_dopant_counts  = npz_file["site_dopant_counts"]

        reader = png.Reader("WSe2_samples/" + str(ind) + "_current_noisy.png")
        pngdata = reader.read()
        px_array = np.array(list(pngdata[2])) / 65535

        return [px_array[None, :, :], np.array([defect_count])]
    else:
        return [None, None]

def one_hot(arr, n_classes):
    encoded = np.zeros((arr.shape[0], n_classes))
    encoded[np.arange(arr.shape[0]), arr.astype(int)] = 1
    return encoded

class MyDataset:
    def __init__(self, train_split_ratio, val_split_ratio, test_split_ratio):
        self.USE_CUDA = True

        print("Loading npz dataset file...")
        npz_file = np.load("WSe2_dataset.npz", allow_pickle = True)
        self.X  = npz_file["X"]
        self.T1 = npz_file["Y1"]
        self.T2 = npz_file["Y2"]
        self.T2 = self.T2[:, :, :, :2]
        print(self.T2.shape)
        self.S  = npz_file["S"]

        img_dtype = torch.float16

        self.X = torch.from_numpy(self.X)  .to(dtype = img_dtype)
        self.T1 = torch.from_numpy(self.T1).to(dtype = torch.float32)
        self.T2 = torch.from_numpy(self.T2).to(dtype = torch.float32)

        if self.USE_CUDA:
            self.X  = self.X.cuda()
            self.T1 = self.T1.cuda()
            self.T2 = self.T2.cuda()

        self.sample_cnt = self.X.shape[0]

        ##Train test split
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio   = val_split_ratio
        self.test_split_ratio  = test_split_ratio

        self.train_sz = int(self.train_split_ratio * self.sample_cnt)
        self.val_sz   = int(self.val_split_ratio   * self.sample_cnt)

        val_ind      = self.train_sz + self.val_sz
        self.test_sz = self.sample_cnt - val_ind

        self.X_train  = self.X [:self.train_sz]
        self.T1_train = self.T1[:self.train_sz]
        self.T2_train = self.T2[:self.train_sz]
        self.X_val    = self.X [self.train_sz:val_ind]
        self.T1_val   = self.T1[self.train_sz:val_ind]
        self.T2_val   = self.T2[self.train_sz:val_ind]
        self.X_test   = self.X [val_ind:]
        self.T1_test  = self.T1[val_ind:]
        self.T2_test  = self.T2[val_ind:]

        self.T1_mean = torch.mean(self.T1_train)
        self.T1_std  = torch.std (self.T1_train)

    def select_target(self, target):
        if target == "count":
            self.Y       = self.T1
            self.Y_train = self.T1_train
            self.Y_val   = self.T1_val
            self.Y_test  = self.T1_test
        elif target == "lattice":
            self.Y       = self.T2
            self.Y_train = self.T2_train
            self.Y_val   = self.T2_val
            self.Y_test  = self.T2_test

    def T_transform(self, T):
        return (T - self.T_mean) / self.T_std

    def T_inverse_transform(self, T):
        return T * self.T_std + self.T_mean

    ## Evaluation
    def evaluate_model(self, model, batch_size):
        def print_metrics(model, X, T, batch_size):
            T_hat = pytorch_utils.predict(model, X, batch_size)

            MSE = torch.mean((T_hat - T) ** 2)
            print("\tMSE:", MSE)
            print("\n\tRMSE:", torch.sqrt(MSE))
            print("\n\tMAE:", torch.mean(torch.abs(T_hat - T)))

        print("Train metrics:")
        print_metrics(model, self.X_train, self.Y_train, batch_size)
        print("\nValidation metrics:")
        print_metrics(model, self.X_val,   self.Y_val  , batch_size)
        print("\nTest metrics:")
        print_metrics(model, self.X_test,  self.Y_test , batch_size)
