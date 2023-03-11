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

## Pytorch Import
import torch 
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# torchmetrics
import torchmetrics

from utils import *

from tqdm.auto import tqdm, trange

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL


################# Train One Epoch() ###################
def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, n_classes, scheduler = None, grad_clipping = False):

    ################ torchmetrics: initialize metric #########################

    metric_acc = torchmetrics.Accuracy(task='multiclass', average = 'macro', num_classes=n_classes,).to(device)
    metric_f1 = torchmetrics.F1Score(task="multiclass", average = 'macro', num_classes=n_classes,).to(device)
    
    ############################################################################

    train_loss = 0
    dataset_size = 0

    bar = tqdm(enumerate(dataloader), total = len(dataloader), desc='Train Loop')
    # bar = tqdm_notebook(enumerate(dataloader), total = len(dataloader), desc='Train Loop', leave=False)

    model.train()
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype = torch.long)
        masks = data['attention_mask'].to(device, dtype = torch.long)

        # targets
        targets = data['target'].to(device, dtype = torch.long)
 
        # y_preds
        y_preds = model(ids, masks) 
        
        # Loss
        loss = loss_fn(y_preds, targets)

        optimizer.zero_grad()
        loss.backward()

        # Gradient-Clipping | source: https://velog.io/@seven7724/Transformer-계열의-훈련-Tricks
        max_norm = 5
        if grad_clipping:
            #print("Gradient Clipping Turned On | max_norm: ", max_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        batch_size = ids.size(0)
        dataset_size += batch_size
        train_loss += float(loss.item() * batch_size) 
        train_epoch_loss = train_loss / dataset_size 
        
        # ACC, F1
        acc = metric_acc(y_preds, targets)
        acc = acc.detach().cpu().item()
        f1 = metric_f1(y_preds, targets)
        f1 = f1.detach().cpu().item()

        bar.set_postfix(Epoch = epoch,  
                        Train_loss = train_epoch_loss,
                        LR = optimizer.param_groups[0]['lr'],                       
                        ACC = acc,
                        F1 = f1,
                        )
        
    # Type - ACC, F1
    train_acc = metric_acc.compute()
    train_f1 = metric_f1.compute()

    print("Train's Accuracy: %.2f | F1_SCORE %.3f" % (train_acc, train_f1))
    print()

    # Reseting internal state such that metric ready for new data
    metric_acc.reset()
    metric_f1.reset()

    torch.cuda.empty_cache()
    _ = gc.collect()

    return train_epoch_loss, train_acc, train_f1
  
  
   
################# Valid One Epoch() ###################
@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, n_classes):

    ################ torchmetrics: initialize metric #########################

    metric_acc = torchmetrics.Accuracy(task='multiclass', average = 'macro', num_classes=n_classes).to(device)
    metric_f1 = torchmetrics.F1Score(task="multiclass", average = 'macro', num_classes=n_classes).to(device)
    
    ############################################################################
    
    valid_loss = 0
    dataset_size = 0
    
    #tqdm의 경우, for문에서 iterate할 때 실시간으로 보여주는 라이브러리입니다. 보시면 압니다. 
    bar = tqdm(enumerate(dataloader), total = len(dataloader), desc='Valid Loop')
    # bar = tqdm_notebook(enumerate(dataloader), total = len(dataloader), desc='Valid Loop', leave=False)

    model.eval()
    with torch.no_grad():
        for step, data in bar:
            ids = data['input_ids'].to(device, dtype = torch.long)
            masks = data['attention_mask'].to(device, dtype = torch.long)

            # targets
            targets = data['target'].to(device, dtype = torch.long)
    
            # y_preds
            y_preds = model(ids, masks) 
            
            # Loss
            loss = loss_fn(y_preds, targets)

            # 실시간 Loss
            batch_size = ids.size(0)
            dataset_size += batch_size
            valid_loss += float(loss.item() * batch_size)
            valid_epoch_loss = valid_loss / dataset_size

            # ACC, F1
            acc = metric_acc(y_preds, targets)
            acc = acc.detach().cpu().item()
            f1 = metric_f1(y_preds, targets)
            f1 = f1.detach().cpu().item()

            bar.set_postfix(Epoch = epoch,  
                            Valid_loss = valid_epoch_loss,
                            LR = optimizer.param_groups[0]['lr'],                       
                            ACC = acc,
                            F1 = f1,
                            )


    # Type - ACC, F1
    valid_acc = metric_acc.compute()
    valid_f1 = metric_f1.compute()

    print("Valid's Accuracy: %.2f | F1_SCORE %.3f" % (valid_acc, valid_f1))
    print()

    # Reseting internal state such that metric ready for new data
    metric_acc.reset()
    metric_f1.reset()

    torch.cuda.empty_cache()
    _ = gc.collect()

    return valid_epoch_loss, valid_acc, valid_f1
  
  
  
################### Run Train ###################
def run_train(model, 
              model_save, 
              train_loader, 
              valid_loader, 
              loss_fn, 
              optimizer, 
              device, 
              n_classes, 
              fold, 
              scheduler = None, 
              grad_clipping = False, 
              n_epochs=5):
    
    if torch.cuda.is_available():
        print("INFO: GPU - {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict()) 
    # inference with models which is saved at best_score, or lowest_loss updated! 
    # Don't Need to save bst_model_wts like above

    lowest_epoch = np.inf
    lowest_loss = np.inf

    train_hs, valid_hs, train_f1s, valid_f1s, train_accs, valid_accs = [], [], [], [], [], []
    
    best_score = 0
    best_score_epoch = np.inf
    best_model = None


    for epoch in range(1, n_epochs +1):
        gc.collect()

        train_epoch_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, n_classes, scheduler, grad_clipping)
        valid_epoch_loss, valid_acc, valid_f1 = valid_one_epoch(model, valid_loader, loss_fn, optimizer, device, epoch, n_classes)
        
        ## 줍줍
        train_hs.append(train_epoch_loss)
        valid_hs.append(valid_epoch_loss)

        train_f1s.append(train_f1)
        valid_f1s.append(valid_f1)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print()
        print(f"Epoch:{epoch:02d} | TL:{train_epoch_loss:.3e} | VL:{valid_epoch_loss:.3e} | Train's Acc: {train_acc:.2f} | Valid's Acc: {valid_acc:.2f} |")
        print()

        if valid_epoch_loss < lowest_loss:
            print(f"{b_}Validation Loss Improved({lowest_loss:.3e}) --> ({valid_epoch_loss:.3e})")
            lowest_loss = valid_epoch_loss
            lowest_epoch = epoch
            # best_model_wts = copy.deepcopy(model.state_dict())
            # PATH = model_save + f"Loss-Fold-{fold}.bin"
            # torch.save(model.state_dict(), PATH)
            # print(f"Better Loss Model Saved{sr_}")

        if best_score < valid_acc:
            print(f"{b_}ACC Improved({best_score:.3f}) --> ({valid_acc:.3f})")
            best_score = valid_acc
            best_score_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH2 = model_save + f"Loss-Fold-{fold}_f1.bin"
            torch.save(model.state_dict(), PATH2)
            print(f"Better_ACC_Model Saved{sr_}")
        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss : %.4e at %d th Epoch of %dth Fold" % (lowest_loss, lowest_epoch, fold))
    print("Best F1(W): %.4f at %d th Epoch of %dth Fold" % (best_score, best_score_epoch, fold))

    # load best model weights
    model.load_state_dict(best_model_wts)

    result = dict()
    result["Train Loss"] = train_hs
    result["Valid Loss"] = valid_hs

    result["Train Acc"] = train_accs
    result["Valid Acc"] = valid_accs

    result["Train F1"] = train_f1s
    result["Valid F1"] = valid_f1s

    # plot
    make_plot(result, stage = "Loss")
    make_plot(result, stage = "Acc")
    make_plot(result, stage = "F1")
    
    del result, train_hs, valid_hs, train_f1s, valid_f1s 

    torch.cuda.empty_cache()
    _ = gc.collect()

    return model, best_score
  
  
 
