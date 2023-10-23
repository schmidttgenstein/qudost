import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.datasets import MNIST
from qudost.multitask.randomproj import *
from scipy.stats import wasserstein_distance
from collections import defaultdict
import random

class CombinedMNIST(Dataset):
    def __init__(self):
        transf = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        train_dataset = MNIST(root='./data', train=True, transform=transf, download=True)
        test_dataset = MNIST(root='./data', train=False, transform=transf, download=True)
        
        train_images = [train_dataset[i][0] for i in range(len(train_dataset))]
        test_images = [test_dataset[i][0] for i in range(len(test_dataset))]

        self.images = torch.stack(train_images + test_images)
        self.labels = torch.cat([train_dataset.targets, test_dataset.targets])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]

class SubDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
    # Add channel dimension
        return self.images[index], self.labels[index]



def split_dataset_with_difference(dataset, difference=0.1):
    # Get a list of indices for each label
    indices_per_label = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        indices_per_label[label.item()].append(idx)

    # Shuffle the indices
    for indices in indices_per_label.values():
        random.shuffle(indices)

    batch1_indices, batch2_indices = [], []

    for label, indices in indices_per_label.items():
        half_len = len(indices) // 2
        diff_amount = int(difference * half_len)
        
        # Add more to batch1 for even digits, more to batch2 for odd digits
        if label % 2 == 0:
            batch1_indices.extend(indices[:half_len + diff_amount])
            batch2_indices.extend(indices[half_len + diff_amount:])
        else:
            batch1_indices.extend(indices[:half_len - diff_amount])
            batch2_indices.extend(indices[half_len - diff_amount:])

    # Shuffle the batches
    random.shuffle(batch1_indices)
    random.shuffle(batch2_indices)

    # Extracting images and labels based on indices
    batch1_images = [dataset.images[i] for i in batch1_indices]
    batch1_labels = [dataset.labels[i] for i in batch1_indices]
    batch2_images = [dataset.images[i] for i in batch2_indices]
    batch2_labels = [dataset.labels[i] for i in batch2_indices]

    # Create SubDataset objects for each batch
    batch1 = SubDataset(batch1_images, batch1_labels)
    batch2 = SubDataset(batch2_images, batch2_labels)

    return batch1, batch2
