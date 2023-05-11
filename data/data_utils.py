import time
import torch 
import numpy as np 
import pandas as pd 
import torch.nn as nn

class DataGenerator:
    def __init__(self,dim = 10,N = 10000):
        self.dim = dim 
        self.N = N 

    def gen_data(self,n = None,split = 0.6,mu_factor = 1):
        if n is None:
            n = self.N 
        x0,y0 = self.gen_label_data(label = 0,n_dat = n,mult = mu_factor)
        x1,y1 = self.gen_label_data(label = 1,n_dat = n,mult = mu_factor)
        cutoff = int(split * x0.shape[1])
        idx = np.arange(n)
        np.random.shuffle(idx)
        x_tr = np.concatenate([x0[:,:cutoff],x1[:,:cutoff]],axis = 1)
        y_tr = np.concatenate([y0[:,:cutoff],y1[:,:cutoff]],axis = 1)
        x_te = np.concatenate([x0[:,cutoff:],x1[:,cutoff:]],axis = 1)
        y_te = np.concatenate([y0[:,cutoff:],y1[:,cutoff:]],axis = 1)
        x_tr = x_tr[:,idx]
        y_tr = y_tr[:,idx]
        np.random.shuffle(idx)
        x_te = x_te[:,idx]
        y_te = y_te[:,idx]
        return x_tr,y_tr,x_te,y_te

    def gen_label_data(self,label = 0, n_dat = None,mult=1):
        if n_dat is None:
            n_dat = self.N 
        y = np.zeros([2,n_dat])
        m = mult * np.random.random(self.dim) - .5 
        C = np.random.random([self.dim,self.dim]) - .5
        C = C @ np.transpose(C)
        x = np.random.multivariate_normal(m, C, n_dat)
        y[label,:] = 1
        return np.transpose(x),y 

    def ym2yo(self,ym):
        vals = 0 * ym
        vals[1,:] = 1
        idx1 = ym == 1
        yo = vals[idx1]
        return yo 

    def probe_dat(self,x,y,coord = 0,show = False):
        yo = self.ym2yo(y)
        idx1 = yo == 1 
        x1 = x[:,idx1]
        x0 = x[:,~idx1]
        if show:
            plt.hist(x0[coord,:],density = True, bins = 100,label = 'x(y=0)')
            plt.hist(x1[coord,:],density = True, bins = 100,label = 'x(y=1)')
            plt.legend()
            plt.show()

class DataSet:
    def __init__(self,x,y, tor = False):
        if tor:
            self.x = torch.tensor(x).float() 
            self.y = torch.tensor(y).int()
        else:
            self.x = x 
            self.y = y
        self.n_samples = x.shape[1]

    def __len__(self):
        return self.n_samples 
    
    def __getitem__(self,idx):
        return self.x[:,idx], self.y[:,idx]

class DataBatcher:
    def __init__(self,dataset,batch_size = 1):
        self.length = dataset.__len__() 
        self.batch_size = batch_size
        self.counter = 0 
        self.dataset = dataset

    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.counter * self.batch_size >= self.length:
            raise StopIteration 
        else:
            c_idx = self.counter 
            bs = self.batch_size 
            start_idx = c_idx * bs 
            end_idx = np.min([(c_idx+1)*bs,self.length])
            self.counter = c_idx + 1
            idxs = np.arange(start_idx,end_idx)
            return self.dataset[idxs]

class DataLoader:
    def __init__(self,dataset,batch_size = 1,tor = False):
        if tor:
            self.dataset = dataset
        else:
            self.dataset = dataset 
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(dataset.__len__() / batch_size))

    def __iter__(self):
        return DataBatcher(self.dataset,self.batch_size)
