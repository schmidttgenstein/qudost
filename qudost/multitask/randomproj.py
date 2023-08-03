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
    def __init__(self, dataset, threshold = None, K=None, p=None, seed=0):
        self.dataset = dataset
        self.n = max(dataset.__getitem__(0)[0].shape) // 2
        self.K = K
        self.p = p
        self.threshold = threshold
        np.random.seed(seed)

    def random_patches(self):
        p_arr = []
        if self.p is None:
            patches = {j: [] for j in np.arange(2, self.n)}
            num_samples = len(self.dataset)
            for _ in range(self.K):
                patch, p = self.generate_patch(num_samples)
                patches[p].append(patch)
                p_arr.append(p)
            patches = {k: v for k, v in patches.items() if len(v) > 0}
            self.p = p_arr
            for j in patches.keys():
                patches[j] = torch.stack(patches[j])
            return patches
        else:
            patches = self.generate_patches_fixed_size()
            return patches

    def eval_separation(self, patch, p):
        patch = patch.unsqueeze(0)
        temp_dataset = Featurization(self.dataset, patch, True, p)
        # Collect feature values for each label
        features_per_label = {i: [] for i in range(10)}  # Assuming MNIST with 10 labels
        for i in range(len(temp_dataset)):
            x, y = temp_dataset[i]
            features_per_label[y].append(x.numpy())
        
        # Compute mean feature value for each label
        mean_features = [np.mean(features_per_label[i]) for i in range(10)]
        
        # Check variance of the mean feature values
        variance = np.var(mean_features)
        return variance

    def generate_patches_fixed_size(self):
        patches = []
        num_samples = len(self.dataset)
        for _ in range(self.K):
            patch = self.generate_patch_fixed_size(num_samples)
            patches.append(patch)
        patches = torch.stack(patches)
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
            
            if patch.std() > 0.0: #check to make sure patches arent all 'background'
                if self.threshold == None:
                    return patch, p
                else:
                    variance = self.eval_separation(patch, p)
                    if variance > self.threshold:
                        return patch, p 

    def generate_patch_fixed_size(self, num_samples):
        while True:
            index = np.random.randint(num_samples)
            image, _ = self.dataset[index]
            _, num_rows, num_cols = image.shape
            top = np.random.randint(num_rows - self.p + 1)
            left = np.random.randint(num_cols - self.p + 1)
            patch = image[:, top:top + self.p, left:left + self.p]
            if patch.std() > 0.0: #check to make sure patches arent all 'background'
                if self.threshold == None:
                    return patch
                else:
                    variance = self.eval_separation(patch, self.p)
                    if variance > self.threshold:
                        return patch 
class Featurization(Dataset):
    def __init__(self, dataset, patches, training=False, p=None):
        self.dataset = dataset
        self.patches = patches
        self.p = p  # Add the 'p' attribute
        self.K = self.calculate_K(patches)
        self.conv_layers = None  # Initialize as None
        self.training = training
        if training:
            self.init_training()


    def calculate_K(self, patches):
        if isinstance(patches, dict):
            return {p: patches[p].shape[0] for p in patches.keys()}  # Variable-sized patches
        else:
            return patches.size(0)  # Fixed-sized patches


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
        if self.conv_layers is None:
            self.conv_layers = self.initialize_conv_layers()

        x_data = {}
        y_data = {}
        for j in range(self.dataset.__len__()):
            x, y = self.dataset[j]
            x_data[j] = self.featurize_input(x)
            y_data[j] = y
        self.x_data = x_data
        self.y_data = y_data

    def initialize_conv_layers(self):
        if isinstance(self.patches, dict):  # Variable-sized patches
            conv_layers = {p: nn.Conv2d(1, self.K[p], kernel_size=p, bias=False) for p in self.patches.keys()}
            for p in self.patches.keys():
                conv_layers[p].weight.data = self.patches[p]
        else:  # Fixed-sized patches
            conv_layers = {self.p: nn.Conv2d(1, self.K, kernel_size=self.p, bias=True)}
            conv_layers[self.p].weight.data = self.patches
        return conv_layers


    def featurize_input(self, x_data):
        with torch.no_grad():
            image_tensor = x_data.unsqueeze(0)
            if self.conv_layers is None:
                self.conv_layers = self.initialize_conv_layers()
            if isinstance(self.patches, dict):  # Variable-sized patches
                conv_through_patches = []
                for p in self.patches.keys():
                    conv_layer = self.conv_layers[p](image_tensor)
                    if len(conv_layer.shape) == 0:
                        conv_layer = torch.tensor([conv_layer.item()])
                    conv_layer = F.relu(conv_layer)
                    fo = conv_layer.mean(dim=(2, 3)).squeeze() / p
                    if len(fo.shape) == 0:
                        fo = torch.tensor([fo])
                    conv_through_patches.append(fo)
                featurized_output = torch.cat(conv_through_patches)
            else:  # Fixed-sized patches
                conv_layer = self.conv_layers[self.p](image_tensor)
                conv_layer = F.relu(conv_layer)
                featurized_output = conv_layer.mean(dim=(2, 3)).squeeze() / self.p
                if len(featurized_output.shape) == 0:
                    featurized_output = torch.tensor([featurized_output])
        return featurized_output
