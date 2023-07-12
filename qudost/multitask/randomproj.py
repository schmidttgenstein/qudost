import os
import time 
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from qudost.data.data_utils import DataGenerator, DataLoader, DataSetFlipLabel

class RandomPatches:
    def __init__(self, dataset, K=None,seed = 0):
        self.dataset = dataset
        self.n  = max(dataset.__getitem__(0)[0].shape)//2
        self.K = K
        np.random.seed(seed)

    def random_patches(self):
        patches = {j:[] for j in np.arange(2,self.n)}
        num_samples = len(self.dataset)
        for _ in range(self.K):
            patch,p = self.generate_patch(num_samples)
            patches[p].append(patch)
        patches = {k:v for k,v in patches.items() if len(v)>0}
        for j in patches.keys():
            patches[j] = torch.stack(patches[j])
        return patches 

    def generate_patch(self, num_samples):
        while True:
            index = np.random.randint(num_samples)
            p = np.random.randint(2,self.n)
            image, _ = self.dataset[index]
            _, num_rows, num_cols = image.shape
            top = np.random.randint(num_rows - p + 1)
            left = np.random.randint(num_cols - p + 1)
            patch = image[:, top:top + p, left:left + p]
            if patch.std() > 0.0: #check to make sure patches arent all 'white'
                return patch, p 

class Featurization(Dataset):
    def __init__(self, dataset, patches,training = False):
        self.dataset = dataset
        self.patches = patches
        self.K = {p: patches[p].shape[0] for p in patches.keys()}
        conv_layers = {p: nn.Conv2d(1, self.K[p], kernel_size=p, bias=False) for p in patches.keys()}
        for p in patches.keys():
            conv_layers[p].weight.data = self.patches[p]
        self.conv_layers = conv_layers
        #self.conv_layers = nn.Conv2d(1, self.K, kernel_size=self.p, bias=False)
        self.training = training
        if training:
            self.init_training()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.training:
            x = self.x_data[idx]
            y = self.y_data[idx]
        else:
            x, y = self.dataset[idx]
            x = self.featurize_input(x)
        return x, y

    def init_training(self):
        x_data = {}
        y_data = {}
        for j in range(self.dataset.__len__()):
            x,y = self.dataset[j]
            x_data[j] = self.featurize_input(x)
            y_data[j] = y 
        self.x_data = x_data
        self.y_data = y_data 


    def featurize_input(self, x_data):
        # Create convolutional layer with randomly initialized weights
        with torch.no_grad():
            image_tensor = x_data.unsqueeze(0)
            conv_through_patches = []
            for p in self.patches.keys():
                conv_layer = self.conv_layers[p](image_tensor)
                if len(conv_layer.shape) == 0:
                    conv_layer = torch.tensor([conv_layer.item()])
                conv_layer = F.relu(conv_layer)
                fo = conv_layer.mean(dim=(2, 3)).squeeze()/p
                if len(fo.shape) == 0:
                    fo = torch.tensor([fo])
                conv_through_patches.append(fo)
            featurized_output = torch.cat(conv_through_patches)
        return featurized_output

    