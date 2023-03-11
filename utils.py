import re
import os
import gc
import time
import random
import string

import copy
from copy import deepcopy

import numpy as np
import pandas as pd

# Utils
from tqdm.auto import tqdm, trange

import matplotlib.pyplot as plt

## Pytorch Import
import torch 
import torch.nn as nn

# from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader


########### Scheduler #################
import torch.optim as optim

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
      
############## Set Seed ################
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
    
############# Data ################
def dacon_competition_data( base_path ):

    train = pd.read_csv(base_path + 'train.csv')
    test = pd.read_csv(base_path + 'test.csv')
    ss = pd.read_csv(base_path + 'sample_submission.csv')

    print("Train Data Shape: ", train.shape)
    print("Test Data Shape: ", test.shape)
    print("Submission Data Shape: ", ss.shape)

    return train, test, ss
    
    
############# make_plot ####################    
def make_plot(result, stage = "Loss"):

    plot_from = 0

    if stage == "Loss":
        trains = 'Train Loss'
        valids = 'Valid Loss'

    elif stage == "Acc":
        trains = "Train Acc"
        valids = "Valid Acc"

    elif stage == "F1":
        trains = "Train F1"
        valids = "Valid F1"

    plt.figure(figsize=(10, 6))
    
    plt.title(f"Train/Valid {stage} History", fontsize = 20)

    ## Modified for converting Type
    if type(result[trains][0]) == torch.Tensor:
        result[trains] = [num.detach().cpu().item() for num in result[trains]]
        result[valids] = [num.detach().cpu().item() for num in result[valids]]

    plt.plot(
        range(0, len(result[trains][plot_from:])), 
        result[trains][plot_from:], 
        label = trains
        )

    plt.plot(
        range(0, len(result[valids][plot_from:])), 
        result[valids][plot_from:], 
        label = valids
        )

    plt.legend()
    if stage == "Loss":
        plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    
    
    
