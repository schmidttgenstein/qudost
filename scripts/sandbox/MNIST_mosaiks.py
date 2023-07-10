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
import time

class Classification(MLPipeline):
    def __init__(self, epochs=10, lr=0.025, K = 50, classes = 2):
        super().__init__(epochs=epochs, lr=lr)
        self.linear = nn.Linear(K, classes)  #Adjust the input size based on the number of patches used and # of classes for task
        self.cel = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.training = True

    def loss(self, y_pred, y_true):
        return self.cel(y_pred,y_true)  #torch.mean((y_pred - y_true) ** 2)

    def forward(self, x):
        x = self.linear(x)
        #if not self.training:
        #x = torch.softmax(x, dim = 1)
        return x

    def backward(self, y_pred, y_true):
        self.optimizer.zero_grad()
        loss = self.loss(y_pred, y_true)
        loss.backward()


    def update(self, grad=None):
        self.optimizer.step()

    def metrics(self, y_pred, y_true):
        with torch.no_grad():
            y_pred = torch.argmax(y_pred,1)  ### not sure what the following computation was doing y_pred[0], y_true
            #y_true = y_true.view(-1, 1).float()  # Reshape y_true to match the shape of y_pred
        return (y_pred == y_true).numpy().mean()

if __name__ == "__main__":
    # Load MNIST dataset
    mnist_train_dataset = MNIST(root='/Users/schmiaj1/Documents/JHU/data', train=True, transform = ToTensor(), download=True)
    mnist_val_dataset = MNIST(root='/Users/schmiaj1/Documents/JHU/data', train=False, transform = ToTensor(), download=True)

    # Random patch parameters
    patch_size = 7
    num_patches = 150
    start_time = time.time()
        # Initialize RandomPatches and generate random patches
    random_patches = RandomPatches(mnist_train_dataset, K=num_patches, p=patch_size)
    patches = random_patches.random_patches()

    # Create featurized dataset
    featurized_train_dataset = Featurization(mnist_train_dataset, patches,True)
    end_time = time.time()
    featurize_time = end_time - start_time
    print("Featurization Time = ", featurize_time, ' seconds')
    featurized_val_dataset = Featurization(mnist_val_dataset, patches,True)


    #flipping_schemes = [None, "parity", "primality", "loops", "mod_3", "mod_4", "mod_3_binary", "mod_4_binary", "0_to_4_binary"]
    flipping_schemes = ["parity"]
    results = []

    for scheme in flipping_schemes:
        # Create a dataset with flipped labels
        train_flipped_labels = DataSetFlipLabel(featurized_train_dataset, scheme)
        val_flipped_labels = DataSetFlipLabel(featurized_val_dataset, scheme)
        # Create a data loader for the dataset
        batch_size = 250
        train_loader = DataLoader(train_flipped_labels, batch_size=batch_size)
        val_loader = DataLoader(val_flipped_labels, batch_size=batch_size)

        # Determine the number of classes based on the scheme
        if scheme in ["parity", "primality", "loops", "mod_3_binary", "mod_4_binary", "0_to_4_binary"]:
            num_classes = 2
        elif scheme == "mod_3":
            num_classes = 3
        elif scheme == "mod_4":
            num_classes = 4
        elif scheme is None:
            num_classes = 10

        ## these steps are to fit the data loader to what our pipeline expects 
        # (in particular, it's iterating over number of batches)
        #train_loader.batch_size = batch_size
        train_loader.num_batches = int(np.ceil(train_flipped_labels.__len__() / batch_size))
        #val_loader.batch_size = batch_size 
        val_loader.num_batches = int(np.ceil(val_flipped_labels.__len__() / batch_size))

        # Create an instance of MLPipeline for classification
        pipeline = Classification(epochs = 50, lr = 0.005, K = num_patches, classes = num_classes)

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





