import os
import gc
import copy
import time

import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange

import numpy as np
import pandas as pd

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from dataloader import *
from model import *


######### test_loader #################
def make_testloader(test,
                    tokenizer, 
                    max_length, 
                    bs,
                    collate_fn
                    ):

    test_ds = MyDataset(test, 
                        tokenizer = tokenizer,
                        max_length =  max_length,
                        mode = "test")

    test_loader = DataLoader(test_ds,
                             batch_size = bs,
                            # num_workers = 2,
                            # pin_memory = True, 
                             collate_fn = collate_fn,
                             shuffle = False, 
                             drop_last= False)
    
    print("TestLoader Completed")
    return test_loader
  
  
############## test_function #######################  
@torch.no_grad()
def test_func(model, dataloader, device):
    preds= []

    model.eval()
    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total = len(dataloader))
        for step, data in bar:
            ids = data['input_ids'].to(device, dtype = torch.long)
            masks = data['attention_mask'].to(device, dtype = torch.long)

            y_preds = model(ids, masks)

            y_preds = model(ids, masks)
            # y_preds = torch.argmax(y_preds, dim = -1)
            preds.append(y_preds.detach().cpu().numpy())

    predictions = np.concatenate(preds, axis= 0)
    gc.collect()
    
    return predictions
    
    
    
    
################## Trained Model paths #################### 
def trained_model_paths(n_folds = config['n_folds'], model_save = config['model_save']):
    print("n_folds: ",n_folds )

    model_paths_f1 = []
    for num in range(0, n_folds):
        model_paths_f1.append(model_save + f"Loss-Fold-{num}_f1.bin")

    print(len(model_paths_f1))
    print(model_paths_f1)
    return model_paths_f1
    
    
############## inference function ####################    
def inference(model_paths, model_name, dataloader, device):

    final_preds = []
    
    for i, path in enumerate(model_paths):
        model = Model(model_name).to(device)
        model.load_state_dict(torch.load(path))
        
        print(f"Getting predictions for model {i+1}")
        preds = test_func(model, dataloader, device)
        final_preds.append(preds)
    
    # 그리고 평균을 내줍니다.
    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds
    
  
  
  
  
  
