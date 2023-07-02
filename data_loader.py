# This class will be used to load the datasets
import os
import torch
import networkx as nx
from dataset import HW3Dataset

PROCESSED = "/home/student/HW3/data/hw3/processed"
RAW = "/home/student/HW3/data/hw3/raw"

class Datasetloader():
    def __init__(self):
        self.data = torch.load('data/hw3/processed/data.pt')
        

        # masks
        self.train_mask_idx = self.data.train_mask
        self.val_mask_idx = self.data.val_mask
        n = max(max(self.train_mask_idx), max(self.val_mask_idx)) + 1
        self.n = n.item()
        self.train_mask_bool = torch.BoolTensor(self.n).fill_(False)
        self.val_mask_bool = torch.BoolTensor(self.n).fill_(False)
        self.train_mask_bool[self.train_mask_idx] = True
        self.val_mask_bool[self.val_mask_idx] = True


def load_dataset():
    data = torch.load('data/hw3/processed/data.pt')
    train_mask_idx = data.train_mask
    val_mask_idx = data.val_mask
    n = max(max(train_mask_idx), max(val_mask_idx)) + 1
    n = n.item()
    train_mask_bool = torch.BoolTensor(n).fill_(False)
    val_mask_bool = torch.BoolTensor(n).fill_(False)
    train_mask_bool[train_mask_idx] = True
    val_mask_bool[val_mask_idx] = True

    data.train_mask_bool = train_mask_bool
    data.val_mask_bool = val_mask_bool
    data.y = data.y.squeeze(1)
    return data