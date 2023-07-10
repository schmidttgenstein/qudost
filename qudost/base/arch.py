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
    def __init__(self,epochs = 10,lr = 0.025):
        super().__init__()
        self.epochs = epochs
        self.lr = lr 

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

    def train_step(self,x_in,y_truth):
        y_score = self.forward(x_in)
        grad = self.backward(y_score,y_truth)
        self.update(grad)
        return y_score.detach()

    def fit(self,train_loader,val_loader = None,printing = False):
        nbatch_tr = train_loader.num_batches 
        nbatch_val = val_loader.num_batches 
        n_batch = np.max([nbatch_tr, nbatch_val])
        m = val_loader.dataset.__len__()
        results_array = np.zeros([self.epochs,4])
        for epoch in range(self.epochs):
            metrics_array = np.zeros([nbatch_tr, 1])
            vmetrics_array = np.zeros([nbatch_val, 1])
            for batch_idx,(x_data,y_data) in enumerate(train_loader):
                y_score = self.train_step(x_data,y_data)
                train_metrics = self.metrics(y_score, y_data)
                metrics_array[batch_idx, 0] = train_metrics

            # Update train_loss and accuracy based on metrics_array
            train_loss = np.mean(metrics_array[:, 0])
            accuracy = np.mean(metrics_array[:, 0])

            
            for batch_idx,(x_data,y_data) in enumerate(val_loader):
                with torch.no_grad():
                    y_score = self.forward(x_data)
                    val_metrics = self.metrics(y_score,y_data)
                vmetrics_array[batch_idx,0] = val_metrics
            if epoch % 1 == 0:
                a,b,c= self.collate_metrics(metrics_array,vmetrics_array)
                results_array[epoch,:] = np.array([epoch,a,b,c])
                print(f"epoch {epoch}, train_loss {train_metrics:.3f}, accuracy {b:.3f}, train/val acc diff {a:.3f}")
        return results_array

    def collate_metrics(self,m_array,vm_array):
        tr_dat = m_array[:,0].mean() 
        te_dat = vm_array[:,0].mean()
        diff = np.abs(tr_dat - te_dat)
        return diff,tr_dat, te_dat
    
