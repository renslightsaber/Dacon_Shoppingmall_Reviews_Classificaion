import numpy as np
import pandas as pd

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader


## MyDataset()
class MyDataset(Dataset):
    def __init__(self, 
                 df, 
                 tokenizer, 
                 max_length, 
                 mode = "train"):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.reviews = df['reviews']
        
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        reviews = self.reviews[index]
        inputs = self.tokenizer.encode_plus(
            reviews,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            # padding='max_length'
            )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
    
        if self.mode == "train":
            y = self.df.new_target[index]
            return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': y}
        else:
            return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],}
          
          
## prepare_loaders()          
def prepare_loader(train, 
                   fold, 
                   tokenizer, 
                   max_length, 
                   bs,
                   collate_fn 
                   ):
    
    train_df = train[train.kfold != fold].reset_index(drop=True)
    valid_df = train[train.kfold == fold].reset_index(drop=True)

    ## train, valid -> Dataset
    train_ds = MyDataset(train_df, 
                            tokenizer = tokenizer ,
                            max_length = max_length,
                            mode = "train")

    valid_ds = MyDataset(valid_df, 
                            tokenizer = tokenizer ,
                            max_length = max_length,
                            mode = "train")
    
    # Dataset -> DataLoader

    train_loader = DataLoader(train_ds,
                              batch_size = bs, 
                              collate_fn=collate_fn, 
                              num_workers = 2,
                              shuffle = True, 
                              pin_memory = True, 
                              drop_last= True)

    valid_loader = DataLoader(valid_ds,
                              batch_size = bs,
                              collate_fn=collate_fn,
                              num_workers = 2,
                              shuffle = False, 
                              pin_memory = True,)
    
    print("DataLoader Completed")
    return train_loader, valid_loader
