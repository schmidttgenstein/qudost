import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from qudost.data.data_utils import DataGenerator, DataSetFlipLabel
from qudost.base.arch import MLPipeline
from qudost.data.label_flipping import *
from qudost.multitask.custompatch import CustomFilterGenerator
from qudost.multitask.classification import Classification
from qudost.multitask.randomproj import Featurization
import matplotlib.pyplot as plt

import pdb 
if __name__ == "__main__":
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # Load MNIST dataset
    mnist_train_dataset = MNIST(root='/Users/schmiaj1/Documents/JHU/data/', train=False, transform=transf, download=False)

   
    patch_size = 7
    shape_type = 'inv_one'
    # center_size = 10  # Adjust the size of the white center

    filter_generator = CustomFilterGenerator(patch_size)
    patches = filter_generator.create_custom_filter(shape_type).unsqueeze(0)
    
    print(patches)
    # Create featurized dataset



    featurized_train_dataset = Featurization(mnist_train_dataset, patches, True, p = patch_size)
    # Collect the featurized values and labels
    values = []
    labels = []

    for i in range(len(featurized_train_dataset)):
        x, y = featurized_train_dataset[i]
        values.append(x.item())
        labels.append(y)
    values = np.array(values) 
    labels = np.array(labels)
    for i in range(10):
        plt.hist(values[labels == i],bins = 50, density = True, alpha =.5, label = f"label {i}")
    plt.legend()
    plt.show()
   # Plotting the histogram
''' 
    plt.figure(figsize=(10, 6))
    for i in range(10):
        label_values = [value for value, label in zip(values, labels) if label == i]
        plt.hist(label_values, bins=50, alpha=0.5, label=f"Label {i}")
    plt.xlabel("Featurized Values")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Featurized Values by Label (Patch Size: {patch_size}, Shape: {shape_type})")
    plt.legend()
    plt.show()'''