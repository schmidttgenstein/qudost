import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from qudost.multitask.randomproj import *
from torch.utils.data import DataLoader
from qudost.data.data_utils import DataGenerator,  DataSetFlipLabel
from qudost.base.arch import MLPipeline
from qudost.data.label_flipping import * 
import time


class Classification(MLPipeline):
    def __init__(self, epochs=10, lr=0.025, K = 50, classes = 2):
        super().__init__(epochs=epochs, lr=lr,print_mod = 5)
        self.linear = nn.Linear(K, classes)
          #Adjust the input size based on the number of patches used and # of classes for task
        #self.linear2 = nn.Linear(int(K/2),classes)
        self.cel = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.training = True

    def loss(self, y_pred, y_true):
        return self.cel(y_pred,y_true)   

    def forward(self, x):
        x = self.linear(x)
        #x = self.linear2(x)
        return x

    def backward(self, y_pred, y_true):
        #return none for gradient because arch (trainstep) expects it
        self.optimizer.zero_grad()
        loss = self.loss(y_pred, y_true)
        loss.backward()
        return None, loss.detach().item()


    def update(self, grad=None):
        self.optimizer.step()

    def metrics(self, y_pred, y_true,loss=-1):
        with torch.no_grad():
            y_pred = torch.argmax(y_pred,1)  
        return np.array([loss,(y_pred == y_true).numpy().mean()])

    def print_update(self,epoch,results):
        print(f"ep {epoch} with tr loss: {results[0]:.3f}, tr acc: {results[1]:.3f}, val acc: {results[3]:.3f}")
