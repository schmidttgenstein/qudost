import numpy as np
from qudost.multitask.randomproj import *
from qudost.data.label_flipping import *
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import random
from qudost import CombinedMNIST, split_dataset_with_difference
if __name__ == "__main__":
    dataset = CombinedMNIST()
    batch1, batch2 = split_dataset_with_difference(dataset, difference=0.5)

    # Random patch parameters, set patch size to None for variable patch size:
    num_patches = 3
    num_histos = 1
    threshold = None
    patch_size = None
    random_patches = RandomPatches(batch1, threshold = threshold, K = num_patches, p = patch_size)
    patches = random_patches.random_patches()
    featurized_batch1 = Featurization(batch1, patches, num_patches, True, p = patch_size)
    featurized_batch2 = Featurization(batch2, patches, num_patches, True, p = patch_size)
    featurized_batch1.__getitem__(0)

    random_plots = random.sample(range(num_patches), num_histos)

    labels = np.zeros(featurized_batch1.__len__())
    x_data = np.zeros([num_patches,featurized_batch1.__len__()])
    for i in range(len(featurized_batch1)):
        x, y = featurized_batch1[i]
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

    featurized_batch2.__getitem__(0)

    random_plots = random.sample(range(num_patches), num_histos)

    labels = np.zeros(featurized_batch2.__len__())
    x_data = np.zeros([num_patches,featurized_batch2.__len__()])
    for i in range(len(featurized_batch2)):
        x, y = featurized_batch2[i]
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


    num_bins = 50
    histograms_batch1 = []
    histograms_batch2 = []

    for j in range(num_patches):
        data_batch1 = [featurized_batch1[i][0][j].item() for i in range(len(featurized_batch1))]
        data_batch2 = [featurized_batch2[i][0][j].item() for i in range(len(featurized_batch2))]
        hist_batch1, _ = np.histogram(data_batch1, bins=num_bins, density=True)
        hist_batch2, _ = np.histogram(data_batch2, bins=num_bins, density=True)
        histograms_batch1.append(hist_batch1)
        histograms_batch2.append(hist_batch2)
    wasserstein_distances = []
    for i in range(num_patches):
        distance = wasserstein_distance(histograms_batch1[i], histograms_batch2[i])
        wasserstein_distances.append(distance)

    threshold_for_difference = 0.1
    num_above_threshold = sum([1 for dist in wasserstein_distances if dist > threshold_for_difference])
    fraction_above_threshold = num_above_threshold / num_patches

    print("Wasserstein Distances:", wasserstein_distances)
    print("Fraction of patches with distance above threshold:", fraction_above_threshold)
