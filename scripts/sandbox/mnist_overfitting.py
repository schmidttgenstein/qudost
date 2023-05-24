import yaml 
import torch 
import numpy as np 
import pandas as pd 
import torch.nn as nn
import numpy.linalg as la 
import torch.optim as optim
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader as TorchDataLoader
from qudost import CNNetTo

if __name__ == "__main__":
    with open("../../configs/u_config.yaml", mode="rb") as file:
        cfg = yaml.load(file,yaml.BaseLoader)
    mnist_root = cfg['mnist_dir']
    train_data = MNIST(root = mnist_root, transform = ToTensor(), train = False, download = False)
    eval_data = MNIST(root = mnist_root, transform = ToTensor(), train = True, download = False)
    n_data = train_data.__len__()
    n_eval = eval_data.__len__()
    y_counts = np.zeros(10)
    weights = np.zeros(n_data)
        

    train_loader = TorchDataLoader(train_data, batch_size = 250)
    eval_loader = TorchDataLoader(eval_data,batch_size = 100)

    eta = .0025
    num_epochs = 25
    model = CNNetTo(epoch = num_epochs,lr = eta) 
    train_loader.num_batches = train_data.__len__() // train_loader.batch_size + 1 
    eval_loader.num_batches = eval_data.__len__() // eval_loader.batch_size + 1
    res_data = model.fit(train_loader,eval_loader,printing = True)
    p = 0.0075
    m = eval_data.__len__()
    eps = 0.0075
    #summarize_performance(res_data,eps,m,save = True, app ='mnist')
