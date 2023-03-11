import re
import os
import gc
import random
import string

import argparse
import ast

import copy
from copy import deepcopy

import torchmetrics
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAccuracy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

## Transforemr Import
from transformers import AutoTokenizer, AutoModel, AdamW, AutoConfig, DataCollatorWithPadding

# Utils
from tqdm.auto import tqdm, trange

import time
from time import sleep

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


from dataloader import *
from new_trainer import *
from model import *
from utils import *
from inference_utils import *


def define():
    p = argparse.ArgumentParser()


    p.add_argument('--base_path', type = str, default = "./data/", help="Data Folder Path")
    
    ## 다른 경로로 하는 것이 맞다
    p.add_argument('--model_save', type = str, default = "./data/", help="Data Folder Path")
    p.add_argument('--sub_path', type = str, default = "./data/", help="Data Folder Path")
    
    p.add_argument('--model', type = str, default = 'beomi/KcELECTRA-base', help="HuggingFace Pretrained Model")
    
    p.add_argument('--n_folds', type = int, default = 5, help="Folds")
    p.add_argument('--n_epochs', type = int, default = 5, help="Epochs")
    
    p.add_argument('--seed', type = int, default = 2022, help="Seed")
    p.add_argument('--train_bs', type = int, default = 64, help="Batch Size")
    
    p.add_argument('--max_length', type = int, default = 128, help="Max Length")
    
    p.add_argument('--ratio', type = float, default = 0.7, help="Ratio of Train, Valid")
    
    p.add_argument('--T_max', type = int, default = 500, help="T_max")
    p.add_argument('--learning_rate', type = float, default = 5e-5, help="lr")
    p.add_argument('--min_lr', type = float, default = 1e-6, help="Min LR")
    p.add_argument('--weight_decay', type = float, default = 1e-6, help="Weight Decay")
    p.add_argument('--device', type = str, default = "cuda", help="CUDA or MPS or CPU?")


    config = p.parse_args()
    return config
  
def main(config):
    
    ## Data
    train, test, ss = dacon_competition_data(base_path = config.base_path)
    
    ## Set Seed
    set_seed(config.seed)
    
    ## Target Encoding
    train_encode = {v: k for k, v in enumerate(train.target.unique())}
    train_inverse = {v: k for k, v in train_encode.items()}
    print("Target Encode Dictionary: ", train_encode)
    print("Target Decode Dictionary: ", train_inverse)
    print()
    print("new_target")
    train['new_target'] = train.target.apply(lambda x: train_encode[x])
    print(train.head())
    
    ## n_classes
    n_classes = train.new_target.nunique()
    print("n_classes: ", n_classes)
    
    ## Drop Unnecessary Columns 
    train.drop(['id', 'target' ], axis =1, inplace = True)
    print(train.shape)
    print(train.head())
    
    test.drop(['id'], axis =1, inplace = True)
    print(test.shape)
    print(test.head())
    
    ## Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    print("Tokenizer Downloaded")

    # Device
    if config.device == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
    elif config.device == "cuda":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
    else:
        device = torch.device("cpu")
            
    ## DataLoader
    test_loader = make_testloader(test, 
                                  tokenizer, 
                                  config.max_length, 
                                  config.train_bs, 
                                  collate_fn= DataCollatorWithPadding(tokenizer=tokenizer))
    
    ## Saved Models Path
    model_paths_f1 = trained_model_paths(n_folds = config.n_folds, 
                                         model_save = config.model_save)
    
    ## Inference
    f1_preds = inference(model_paths_f1, 
                         config.model, 
                         test_loader,
                         device)
    print("Inference Completed")

    
    # F1 preds
    print("F1 Preds Shape: ", f1_preds.shape) 
    
    # Argmax
    new_preds = np.argmax(f1_preds , axis = 1) # argmax로 target class 뽑아줍니다.
    print("After Argmax, F1 Preds Shape: ", new_preds.shape)
    print()
    
    ## Submission File
    print(ss.shape)
    print(ss.head())
    print()
    
    ## Insert Prediction
    ss['target'] = new_preds
    print(ss.head())
    print()
    print("Value Counts()")
    print(ss.Target.value_counts())
    print()
    
    ## Encoded Target을 원래대로 돌립니다. 
    ss['target'] = ss.Target.apply(lambda x: train_inverse[x])
    ## 확인
    print(ss.shape)
    print(ss.head())
    print()
    print("Value Counts()")
    print(ss.Target.value_counts())
    print()
    
    
    ## Submission csv file name and save
    sub_file_name = config.sub_path + "_".join(config.model.split("/")) +  "_folds_" + str(config.n_folds) + "_epochs_" + str(config.n_epochs) + ".csv"
    print(sub_file_name)
    ss.to_csv(sub_file_name, index=False)
    print("Save Submission.csv")
    

if __name__ == '__main__':
    config = define()
    main(config)
    
    
    
    
