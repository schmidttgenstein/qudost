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
    def __init__(self, dataset, K=None, p=None,seed = 0):
        self.dataset = dataset
        self.K = K
        self.p = p
        np.random.seed(seed)

    def random_patches(self):
        patches = []
        num_samples = len(self.dataset)
        for _ in range(self.K):
            patch = self.generate_patch(num_samples)
            patches.append(patch)
        return torch.stack(patches)

    def generate_patch(self, num_samples):
        while True:
            index = np.random.randint(num_samples)
            image, _ = self.dataset[index]
            _, num_rows, num_cols = image.shape

            top = np.random.randint(num_rows - self.p + 1)
            left = np.random.randint(num_cols - self.p + 1)
            patch = image[:, top:top + self.p, left:left + self.p]

            if torch.any(patch != 0.0): #check to make sure patches arent all 'white'
                return patch

class Featurization(Dataset):
    def __init__(self, dataset, patches,training = False):
        self.dataset = dataset
        self.patches = patches
        self.K = patches.shape[0]
        self.p = patches.shape[-1]
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
            conv_layer = nn.Conv2d(1, self.K, kernel_size=self.p, bias=False)
            conv_layer.weight.data = self.patches
            image_tensor = x_data.unsqueeze(0)
            conv_output = conv_layer(image_tensor)
            conv_output = F.relu(conv_output)
            featurized_output = conv_output.mean(dim=(2, 3)).squeeze()
        return featurized_output

    