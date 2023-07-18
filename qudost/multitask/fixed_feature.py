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
'''
def featurize_dataset(dataset, patch):
    featurized_images = []
    labels = []

    # Iterate over the dataset
    for image, label in dataset:
        # Perform convolution
        output = F.conv2d(image.unsqueeze(0), patch.unsqueeze(0))

        # Apply ReLU activation
        output = F.relu(output)

        # Take the mean of the output
        mean_output = torch.mean(output)

        # Append the featurized image and label to the lists
        featurized_images.append(mean_output)
        labels.append(label)

    # Create a custom dataset with featurized images and labels
    featurized_dataset = CustomDataset(featurized_images, labels)

    return featurized_dataset


class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.images)

'''
if __name__ == "__main__":
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # Load MNIST dataset
    mnist_train_dataset = MNIST(root='./data', train=True, transform=transf, download=True)

   
    patch_size = 3
    shape_type = 'u_shape'
    # center_size = 10  # Adjust the size of the white center

    filter_generator = CustomFilterGenerator(patch_size)
    patches = filter_generator.create_custom_filter(shape_type)
    print(patches)
    # Create featurized dataset



    featurized_train_dataset = Featurization(mnist_train_dataset, patches, True, p = patch_size)
    # Collect the featurized values and labels
    values = []
    labels = []

    for i in range(len(featurized_train_dataset)):
        x, y = featurized_train_dataset[i]
        values.append(x)
        labels.append(y)

   # Plotting the histogram
    plt.figure(figsize=(10, 6))
    for i in range(10):
        label_values = [value for value, label in zip(values, labels) if label == i]
        plt.hist(label_values, bins=50, alpha=0.5, label=f"Label {i}")
    plt.xlabel("Featurized Values")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Featurized Values by Label (Patch Size: {patch_size}, Shape: {shape_type})")
    plt.legend()
    plt.show()