import os
import torch
import numpy as np
import torch.nn as n
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from qudost.multitask.randomproj import *
from torch.utils.data import DataLoader
from qudost.data.data_utils import DataGenerator,  DataSetFlipLabel
from qudost.base.arch import MLPipeline
from qudost.data.label_flipping import *
from qudost.multitask.classification import Classification 
import time
import pickle
import matplotlib.pyplot as plt
import yaml


if __name__ == "__main__":
<<<<<<< HEAD
    dataset = 'MNIST' #CIFAR10 or MNIST
=======
    dataset = 'CIFAR10' #CIFAR10 or MNIST
   
>>>>>>> dadaabbe4ec8bf2993320ce2c1af04936582091e
    # Load  dataset
    #mnist_train_dataset = MNIST(root='/Users/schmiaj1/Documents/JHU/data/', train=True, transform = transf, download=True)
    #mnist_val_dataset = MNIST(root='/Users/schmiaj1/Documents/JHU/data/', train=False, transform =transf, download=True)
    #mnist_train_dataset = MNIST(root='C:\\Users\\juand\\OneDrive - Johns Hopkins\\JHU\\2023.Summer\\James Research\\data\\', train=True, transform = transf, download=True)
    #mnist_val_dataset = MNIST(root='C:\\Users\\juand\\OneDrive - Johns Hopkins\\JHU\\2023.Summer\\James Research\\data\\', train=False, transform =transf, download=True)
    transf = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

    if dataset == 'CIFAR10':
        train_dataset = CIFAR10(root='./data', train=True, transform = transf, download=True)
        val_dataset = CIFAR10(root='./data', train=False, transform =transf, download=True)
    elif dataset == 'MNIST':
        train_dataset = MNIST(root='./data', train=True, transform = transf, download=True)
        val_dataset = MNIST(root='./data', train=False, transform =transf, download=True)
    # Random patch parameters, set patch size to None for variable patch size:
    num_patches = 2
    num_histos = 2
    threshold = None
    patch_size = None
    start_time = time.time()
        # Initialize RandomPatches and generate random patches
    random_patches = RandomPatches(train_dataset, threshold = threshold, K = num_patches, p = patch_size)
    patches = random_patches.random_patches()
    '''
    def zca_whitening(patches):
    # Handle the case when patches are a dictionary
        if isinstance(patches, dict):
            whitened_patches = {}
            for key in patches.keys():
                patch_tensor = patches[key]
                whitened_patch_tensor = _whiten_patches(patch_tensor)
                whitened_patches[key] = whitened_patch_tensor
            return whitened_patches
        # Handle the case when patches are stacked in a tensor
        else:
            return _whiten_patches(patches)

    def _whiten_patches(patch_tensor):
        # Flatten patches
        original_shape = patch_tensor.shape
        flattened_patches = patch_tensor.view(original_shape[0], -1).numpy()
        
        # Compute the covariance matrix
        cov_matrix = np.cov(flattened_patches, rowvar=False)
        
        # Perform SVD on the covariance matrix
        U, S, _ = np.linalg.svd(cov_matrix)
        
        # Compute the whitened patches
        epsilon = 1e-5  # To avoid division by zero
        whitening_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
        whitened_patches = np.dot(flattened_patches, whitening_matrix.T)
        
        # Reshape whitened patches back to their original shape
        whitened_patch_tensor = torch.tensor(whitened_patches).view(original_shape)
        
        return whitened_patch_tensor
    patches = zca_whitening(patches)
   
    def zca_whitening(patches):
        # Flatten the patches
        patches_flat = patches.reshape(patches.shape[0], -1)
        
        # Compute mean and center the data
        mean = patches_flat.mean(axis=0)
        patches_centered = patches_flat - mean
        
        # Compute covariance matrix
        covariance = np.cov(patches_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # Add a small constant for numerical stability
        epsilon = 1e-5
        
        # Compute the whitening matrix
        whitening_matrix = np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues + epsilon)))
        
        # Whitening transformation
        patches_whitened_flat = np.dot(patches_centered, whitening_matrix)
        
        # Reshape the whitened data to original shape
        patches_whitened = patches_whitened_flat.reshape(patches.shape)
        
        return patches_whitened
    whitened_patches = zca_whitening(patches)
    '''
    # Create featurized dataset
    featurized_train_dataset = Featurization(train_dataset, patches, num_patches, True, p = patch_size)
    featurized_val_dataset = Featurization(val_dataset, patches, num_patches, True, p = patch_size)
    end_time = time.time()
    featurize_time = end_time - start_time
    print("Featurization Time = ", featurize_time, ' seconds')
    with open("train.pkl", "wb") as f:
        pickle.dump(featurized_train_dataset, f)

    with open("test.pkl", "wb") as f:
        pickle.dump(featurized_val_dataset, f)

    with open("ptchs.pkl","wb") as f: 
        pickle.dump(patches,f)

    scheme = None
    featurized_train_dataset.__getitem__(0)

    random_plots = random.sample(range(num_patches), num_histos)

    labels = np.zeros(featurized_train_dataset.__len__())
    x_data = np.zeros([num_patches,featurized_train_dataset.__len__()])
    for i in range(len(featurized_train_dataset)):
        x, y = featurized_train_dataset[i]
        x_data[:,i] = x.numpy()
        labels[i] = y 
    labels = labels.astype(int)
    for j in random_plots:
        plt.figure(j)
        for i in range(10):
            plt.hist(x_data[j,labels == i],bins = 50, density = True, alpha =.5, label = f"label {i}")
        #plt.title(f"feature {j}, patch size {random_patches.p[j]}")
        plt.legend()
    plt.show()

