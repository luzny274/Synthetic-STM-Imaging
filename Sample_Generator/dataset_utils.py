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


class MyDataset:
    def __init__(self, train_split_ratio, val_split_ratio, test_split_ratio, X, Y, USE_CUDA):
        self.USE_CUDA = USE_CUDA

        img_dtype = torch.uint8

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

        if self.USE_CUDA:
            self.X = self.X.cuda()
            self.Y = self.Y.cuda()

        self.sample_cnt = self.X.shape[0]

        ##Train test split
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio   = val_split_ratio
        self.test_split_ratio  = test_split_ratio

        self.train_sz = int(self.train_split_ratio * self.sample_cnt)
        self.val_sz   = int(self.val_split_ratio   * self.sample_cnt)

        val_ind      = self.train_sz + self.val_sz
        self.test_sz = self.sample_cnt - val_ind

        def train_test_split(arr):
            return arr[:self.train_sz], arr[self.train_sz:val_ind], arr[val_ind:] 

        self.X_train, self.X_val, self.X_test = train_test_split(self.X)
        self.Y_train, self.Y_val, self.Y_test = train_test_split(self.Y)

    def transform_initialize():
        self.Y_mean = torch.mean(self.Y_train)
        self.Y_std  = torch.std (self.Y_train)

    def T_transform(self, T):
        return (T - self.Y_mean) / self.Y_std

    def T_inverse_transform(self, T):
        return T * self.Y_std + self.Y_mean

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
