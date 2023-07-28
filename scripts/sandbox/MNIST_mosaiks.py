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

if __name__ == "__main__":

    transf = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    # Load MNIST dataset
    mnist_train_dataset = MNIST(root='/Users/schmiaj1/Documents/JHU/data/', train=True, transform = transf, download=True)
    mnist_val_dataset = MNIST(root='/Users/schmiaj1/Documents/JHU/data/', train=False, transform =transf, download=True)

    # Random patch parameters, set patch size to None for variable patch size:
    num_patches = 50
    patch_size = None
    start_time = time.time()
        # Initialize RandomPatches and generate random patches
    random_patches = RandomPatches(mnist_train_dataset, K = num_patches, p = patch_size)
    patches = random_patches.random_patches()

    # Create featurized dataset
    featurized_train_dataset = Featurization(mnist_train_dataset, patches, True, p = patch_size)
    featurized_val_dataset = Featurization(mnist_val_dataset, patches, True, p = patch_size)
    end_time = time.time()
    featurize_time = end_time - start_time
    print("Featurization Time = ", featurize_time, ' seconds')

    #flipping_schemes = [None, "parity", "primality", "loops", "mod_3", "mod_4", "mod_3_binary", "mod_4_binary", "0_to_4_binary"]
    flipping_schemes = ["squash_4"]
    results = []

    for scheme in flipping_schemes:
        # Create a dataset with flipped labels
        print(scheme)
        train_flipped_labels = DataSetFlipLabel(featurized_train_dataset, scheme)
        val_flipped_labels = DataSetFlipLabel(featurized_val_dataset, scheme)
        # Create a data loader for the dataset
        batch_size = 150
        train_loader = DataLoader(train_flipped_labels, batch_size=batch_size)
        val_loader = DataLoader(val_flipped_labels, batch_size=batch_size)

        # Determine the number of classes based on the scheme
        if scheme in ["parity", "primality", "loops", "mod_3_binary", "mod_4_binary", "0_to_4_binary"]:
            num_classes = 2
        elif scheme in ["squash_4","mod_3"]:
            num_classes = 3
        elif scheme in  ["squash_3","mod_4"]:
            num_classes = 4
        elif scheme == "plus_1":
            num_classes = 10
        elif scheme is None:
            num_classes = 10

        ## these steps are to fit the data loader to what our pipeline expects 
        # (in particular, it's iterating over number of batches)
        #train_loader.batch_size = batch_size
        train_loader.num_batches = int(np.ceil(train_flipped_labels.__len__() / batch_size))
        #val_loader.batch_size = batch_size 
        val_loader.num_batches = int(np.ceil(val_flipped_labels.__len__() / batch_size))

        # Create an instance of MLPipeline for classification
        pipeline = Classification(epochs = 150, lr = 0.025, K = num_patches, classes = num_classes)

        # Fit the MLPipeline on the dataset
        results_array = pipeline.fit(train_loader, val_loader,printing = True)

        # Append the results to the list
        results.append((scheme, results_array))

    # Print the results
    for scheme, results_array in results:
        print(f"Results for flipping scheme: {scheme}")
        print(results_array)
        print("")

    # Save the results to a file
    output_dir = "./linear_regression_output"
    os.makedirs(output_dir, exist_ok=True)

    for scheme, results_array in results:
        file_path = os.path.join(output_dir, f"results_{scheme}.txt")
        np.savetxt(file_path, results_array)





