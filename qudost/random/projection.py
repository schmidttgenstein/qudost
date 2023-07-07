import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from qudost.density import ECDF, EPDF, RegressionCDF, DensityNetwork
from qudost.data import DataSet, DataLoader, ImageColorProj


def load_cifar(download = True):
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR100(root='.', train=True, transform=transform, download=download)
    return dataset

if __name__ == "__main__":
    idx = 0
    proj = ImageColorProj(load_cifar(download=True))
    print("Sum R value:", proj.r_proj_getitem(idx=idx, dim=0))
    print("Sum G value:", proj.r_proj_getitem(idx=idx, dim=1))
    print("Sum B value:", proj.r_proj_getitem(idx=idx, dim=2))