import torch
from torchvision import datasets, transforms
from qudost.multitask.randomproj import Featurization, RandomPatches
import json

def load_cifar(download = True):
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR100(root='.', train=True, transform=transform, download=download)
    return dataset

if __name__ == "__main__":

    '''
    idx = 0
    proj = ImageColorProj(load_cifar(download=True))
    print("Sum R value:", proj.r_proj_getitem(idx=idx, dim=0))
    print("Sum G value:", proj.r_proj_getitem(idx=idx, dim=1))
    print("Sum B value:", proj.r_proj_getitem(idx=idx, dim=2))
    '''
    transf = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    # Load MNIST dataset
    mnist_train_dataset = datasets.MNIST(root='C:\\Users\\juand\\OneDrive - Johns Hopkins\\JHU\\2023.Summer\\James Research\\datasets\\', train=True, transform = transf, download=True)
    mnist_val_dataset = datasets.MNIST(root='C:\\Users\\juand\\OneDrive - Johns Hopkins\\JHU\\2023.Summer\\James Research\\datasets\\', train=False, transform =transf, download=True)

    # Random patch parameters
    num_patches = 150
    # Initialize RandomPatches and generate random patches
    random_patches = RandomPatches(mnist_val_dataset, K=num_patches)
    patches = random_patches.random_patches()

    # Create featurized dataset
    #featurized_dataset = Featurization(mnist_train_dataset, patches,True)
    featurized_dataset = Featurization(mnist_val_dataset, patches,True)

    feature_values_by_patches = {}

    for i in range(num_patches):
        collection = {}
        for j in range(featurized_dataset.__len__()):
            item = featurized_dataset[j]
            tensor = item[0]  
            label = item[1]
            if label not in collection:
                collection[label] = []
            collection[label].append(tensor[i])
        # Append the i-th entry of each tensor to the collection
    
        feature_values_by_patches[i] = {label: torch.stack(tensors).detach().numpy().tolist() for label, tensors in collection.items()}

    with open('feature_values_by_patches.json', 'w') as file:
        json.dump(feature_values_by_patches, file, indent=4)
    



