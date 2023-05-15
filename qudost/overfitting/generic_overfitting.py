import time
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
#from torch.utils.data import DataLoader as TorchDataLoader


from overfitting import CNNetTo, FCNetTo, FCNetFS
from data import DataGenerator, DataSet, DataLoader

if __name__ == "__main__":
    
    input_dim = 20
    amod_list = [] 
    bmod_list = []
    tr_data_list = []
    val_dat_list = []

        

    N = 50000
    bs = N // 5
    num_epochs = 50
    eta = 0.01
    dg = DataGenerator(dim = input_dim,N = N)
    xtr,ytr,xval,yval = dg.gen_data(mu_factor = 1,split=.5)
    ds_tr = DataSet(xtr,ytr)
    ds_val = DataSet(xval,yval)
    dl_tr = DataLoader(ds_tr,bs)
    dl_val = DataLoader(ds_val, bs)
   
    ds_tr_to = DataSet(xtr,ytr,tor = True)
    ds_val_to = DataSet(xval,yval,tor = True)
    dl_tr_to = DataLoader(ds_tr_to,5000)
    dl_val_to = DataLoader(ds_val_to, 5000)
    num_epochs = 50
    cnet = FCNetTo(dims = [input_dim,25,15,20,10,2], epoch = num_epochs, lr=eta)
    t0 = time.time()
    r_dat = cnet.fit(dl_tr_to,dl_val_to,printing = True)

    eps = 0.0075
    #summarize_performance(r_dat,eps,N,save = True, app = 'fc net')
    t1 = time.time()-t0 
    print('-------------')
    print(f"took {t1:3f} seconds to run {num_epochs} epochs")
    print('-------------')  
    time.sleep(1)

