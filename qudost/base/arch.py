import time
import torch 
import numpy as np 
import pandas as pd 
import torch.nn as nn
import numpy.linalg as la 
import torch.optim as optim
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader as TorchDataLoader


class MLPipeline(nn.Module):
    def __init__(self,epochs = 10,lr = 0.025,print_mod = 1):
        super().__init__()
        self.epochs = epochs
        self.lr = lr 
        self.print_mod = print_mod

    def loss(self,):
        raise NotImplementedError

    def forward(self,):
        raise NotImplementedError

    def backward(self,):
        raise NotImplementedError 

    def update(self,):
        raise NotImplementedError  

    def metrics(self,x,y):
        raise NotImplementedError
    
    def print_update(self,epoch,results):
        raise NotImplementedError

    def train_step(self,x_in,y_truth):
        y_score = self.forward(x_in)
        grad, loss = self.backward(y_score,y_truth)
        self.update(grad)
        return y_score.detach(),loss 

    def fit(self,train_loader,val_loader = None,printing = False): 
        results_array = np.array([])
        tmetrics_array = np.array([])
        vmetrics_array = np.array([])
        for epoch in range(self.epochs):
            for _,(x_data,y_data) in enumerate(train_loader):
                y_score,loss = self.train_step(x_data,y_data)
                train_metrics = self.metrics(y_score, y_data,loss)
                tmetrics_array = np.vstack([tmetrics_array,train_metrics]) if tmetrics_array.size else  train_metrics
            if val_loader is not None:
                for _,(x_data,y_data) in enumerate(val_loader):
                    with torch.no_grad():
                        y_score = self.forward(x_data)
                        val_metrics = self.metrics(y_score,y_data)
                    vmetrics_array  = np.vstack([vmetrics_array,val_metrics]) if vmetrics_array.size else val_metrics
                results = np.concatenate([tmetrics_array.mean(axis = 0),vmetrics_array.mean(axis = 0)])
            else:
                results = tmetrics_array.mean(axis = 0)
            results_array = np.vstack([results_array,results]) if results_array.size else results
            if epoch % self.print_mod == 1:
                self.print_update(epoch,results)
        return results_array