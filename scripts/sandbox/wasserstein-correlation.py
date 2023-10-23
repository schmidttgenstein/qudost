import numpy as np
from qudost.multitask.randomproj import *
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from qudost import CombinedMNIST, split_dataset_with_difference

if __name__ == "__main__":
    # Set number of patches
    num_patches = 2
    patch_size = None
    num_bins = 50
    dataset = CombinedMNIST()
    threshold_for_difference = 0.1
    # Set difference values
    difference_values = [i * 0.1 for i in range(3)]

    # Results storage
    fractions_above_threshold = []

    # Iterate over each difference value
    for difference in difference_values:
        # Split the dataset with current difference value
        batch1, batch2 = split_dataset_with_difference(dataset, difference=difference)
        
        # Generate patches and featurize batches
        random_patches = RandomPatches(batch1, threshold=None, K=num_patches, p=patch_size)
        patches = random_patches.random_patches()
        featurized_batch1 = Featurization(batch1, patches, num_patches, True, p=patch_size)
        featurized_batch2 = Featurization(batch2, patches, num_patches, True, p=patch_size)
        
        # Compute histograms for batches
        histograms_batch1, histograms_batch2 = [], []
        for j in range(num_patches):
            data_batch1 = [featurized_batch1[i][0][j].item() for i in range(len(featurized_batch1))]
            data_batch2 = [featurized_batch2[i][0][j].item() for i in range(len(featurized_batch2))]
            hist_batch1, _ = np.histogram(data_batch1, bins=num_bins, density=True)
            hist_batch2, _ = np.histogram(data_batch2, bins=num_bins, density=True)
            histograms_batch1.append(hist_batch1)
            histograms_batch2.append(hist_batch2)
        
        # Compute Wasserstein distances
        wasserstein_distances = []
        for i in range(num_patches):
            distance = wasserstein_distance(histograms_batch1[i], histograms_batch2[i])
            wasserstein_distances.append(distance)
        
        # Compute fraction of patches with Wasserstein distance above the threshold
        num_above_threshold = sum([1 for dist in wasserstein_distances if dist > threshold_for_difference])
        fraction_above_threshold = num_above_threshold / num_patches
        
        # Store result
        fractions_above_threshold.append(fraction_above_threshold)

    # Plot results
    plt.plot(difference_values, fractions_above_threshold, marker='o')
    plt.xlabel("Difference Value")
    plt.ylabel("Fraction Above Threshold")
    plt.title("Fraction of Patches with Wasserstein Distance Above Threshold vs. Difference Value")
    plt.grid(True)
    plt.show()



