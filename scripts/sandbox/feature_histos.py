import os
import torch
import numpy as np
import torch.nn as n
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from qudost.multitask.randomproj import *
from torch.utils.data import DataLoader
from qudost.data.data_utils import DataGenerator,  DataSetFlipLabel
from qudost.base.arch import MLPipeline
from qudost.data.label_flipping import *
from qudost.multitask.classification import Classification 
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":

    transf = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    # Load MNIST dataset
    #mnist_train_dataset = MNIST(root='/Users/schmiaj1/Documents/JHU/data/', train=True, transform = transf, download=True)
    mnist_val_dataset = MNIST(root='/Users/schmiaj1/Documents/JHU/data/', train=False, transform =transf, download=False)

    # Random patch parameters, set patch size to None for variable patch size:
    num_patches = 15
    patch_size = None
    start_time = time.time()
        # Initialize RandomPatches and generate random patches
    random_patches = RandomPatches(mnist_val_dataset, K = num_patches, p = patch_size)
    patches = random_patches.random_patches()

    val_dataset = Featurization(mnist_val_dataset, patches, True, p = patch_size)
    
    #flipping_schemes = [None, "parity", "primality", "loops", "mod_3", "mod_4", "mod_3_binary", "mod_4_binary", "0_to_4_binary"]
    scheme = "primality"
    featurized_val_dataset = DataSetFlipLabel(val_dataset, scheme)
    featurized_val_dataset.__getitem__(0)
    
    labels = np.zeros(featurized_val_dataset.__len__())
    x_data = np.zeros([num_patches,featurized_val_dataset.__len__()])
    for i in range(len(featurized_val_dataset)):
        x, y = featurized_val_dataset[i]
        x_data[:,i] = x.numpy()
        labels[i] = y 
    labels = labels.astype(int)
    for j in range(num_patches):
        plt.figure(j)
        for i in range(10):
            plt.hist(x_data[j,labels == i],bins = 50, density = True, alpha =.5, label = f"label {i}")
        plt.title(f"feature {j}, patch size {random_patches.p[j]}")
        #plt.legend()
    plt.show()