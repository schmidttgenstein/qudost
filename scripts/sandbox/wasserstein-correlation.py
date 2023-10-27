import numpy as np
from qudost.multitask.randomproj import *
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from qudost import CombinedMNIST, split_featurized_dataset
import pickle
import time
if __name__ == "__main__":
    
    # Hyperparameters
    num_patches = 1
    patch_size = 5
    # Initialize the dataset and random patches
    dataset = CombinedMNIST()
    random_patches = RandomPatches(dataset, threshold=None, K=num_patches, p=patch_size)
    patches = random_patches.random_patches()
    featurized_dataset= Featurization(dataset, patches, num_patches, True, p=patch_size)
    

    N = 1000
    with open("featurized_MNIST_10k-p=5.pkl", "rb") as f:
        featurized_dataset = pickle.load(f)

    # Modify the datasets to keep only the first N features
    for idx in featurized_dataset.x_data:
        featurized_dataset.x_data[idx] = featurized_dataset.x_data[idx][:N]
    num_patches = N 
    num_bins = 50
    difference_values = [i * 0.5 for i in range(10)]
    # Results storage
    mean_wasserstein_distances = []

    # Iterate over each difference value
    for difference in difference_values:
        batch_start_time = time.time()
        # Split the dataset with current difference value
        batch1, batch2 = split_featurized_dataset(featurized_dataset, difference=difference)
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        print("Batch Time = ", batch_time, ' seconds')
        
        wass_start_time = time.time()
        # Compute Wasserstein distances directly between the batches' entries
        wasserstein_distances = [wasserstein_distance(batch1[i][0], batch2[i][0]) for i in range(num_patches)]
        wass_end_time = time.time()
        wass_time = wass_end_time-wass_start_time
        print("Wasserstein Distance Time = ", wass_time, ' seconds')
        # Compute the mean Wasserstein distance for the current difference value
        mean_distance = np.mean(wasserstein_distances)
        
        # Store result
        mean_wasserstein_distances.append(mean_distance)

    # Plot results
    plt.plot(difference_values, mean_wasserstein_distances, marker='o')
    plt.xlabel("Difference Value")
    plt.ylabel("Mean Wasserstein Distance")
    plt.title("Mean Wasserstein Distance vs. Difference Value")
    plt.grid(True)
    plt.show()