import time
import torch 
import numpy as np 
import pandas as pd 
import torch.nn as nn
from torchvision import datasets, transforms
from qudost.data.label_flipping import * 

class DataGenerator:
    def __init__(self,dim = 10,N = 10000, n_mixture = 2, split = 0.5, tor = False):
        self.dim = dim 
        self.N = N 
        self.n_mixture = n_mixture
        self.split = split
        self.tor = tor

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

    def train_test_split(self, x):
        np.random.shuffle(x)
        split_idx = int(self.split * x.shape[0])
        x_tr = x[:split_idx]
        x_te = x[split_idx:]
        if self.tor: 
            x_tr = torch.tensor(x_tr)
            x_te = torch.tensor(x_te)
        return x_tr, x_te

    def fat_tail(self, thick = 0.5, trim = 20):
        n = 1/thick
        y = np.random.random(self.N)
        u = np.zeros(y.shape[0])
        idx1 =  y<=1/n
        u[idx1] =  -1/(n*y[idx1])
        idx2 = (y>1/n) & (y<=(n-1)/n)
        u[idx2] = n*y[idx2]-(n-2)
        idx3 = y>(n-1)/n
        u[idx3] = 1/(n*(1-y[idx3]))
        u = u[(u>-trim)&(u<trim)]
        return self.train_test_split(u)
    
    def gmm_data(self,):
        datas = []
        for j in range(self.n_mixture):
            m = np.random.normal(0,5)
            s = np.random.lognormal()
            data = np.random.normal(m,s,self.N)
            datas.append(data)
        data = np.concatenate(datas)
        return self.train_test_split(data)
    
    def cauchy(self, trim = 20):
        s = np.random.standard_cauchy(self.N)
        s = s[(s>-trim) & (s<trim)]
        return self.train_test_split(s)
        
class DataSet:
    def __init__(self,x,y,tor = False,zdim = False):
        self.zdim = zdim 
        if tor:
            self.x = torch.tensor(x).float() 
            self.y = torch.tensor(y).float()
        else:
            self.x = x
            self.y = y
        if zdim:
            self.n_samples = x.shape[0]
        else:
            self.n_samples = x.shape[1]

    def __len__(self):
        return self.n_samples 
    
    def __getitem__(self,idx):
        if self.zdim:
            return self.x[idx], self.y[idx]
        else:
            return self.x[:,idx], self.y[:,idx]

class DataBatcher:
    def __init__(self,dataset,batch_size = 1):
        self.length = dataset.__len__() 
        self.batch_size = batch_size
        self.counter = 0 
        self.dataset = dataset
        #self.zdim = dataset.zdim

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



class DataSetFlipLabel:
    def __init__(self, dataset, scheme=None):
        self.orig_dataset = dataset
        self.scheme = scheme

    def flip_label(self, y):
        if self.scheme is None:
            return y  # No scheme specified, leave label as is
        elif self.scheme == "parity":
            return flip_parity_label(y)
        elif self.scheme == "primality":
            return flip_primality_label(y)
        elif self.scheme == "loops":
            return flip_loop_label(y)
        elif self.scheme == "mod_3":
            return flip_mod_3_label(y)
        elif self.scheme == "mod_4":
            return flip_mod_4_label(y)
        elif self.scheme == "mod_3_binary":
            return flip_mod_3_binary_label(y)
        elif self.scheme == "mod_4_binary":
            return flip_mod_4_binary_label(y)
        elif self.scheme == "0_to_4_binary":
            return flip_0_to_4_binary_label(y)
        elif self.scheme == "plus_1":
            return plus_1(y)
        elif self.scheme == "squash_3":
            return squash_3(y)
        elif self.scheme == "squash_4":
            return squash_4(y)
        elif self.scheme == "find_1":
            return find_1(y)
        else:
            return y
    
    def __len__(self):
        return self.orig_dataset.__len__() 
    
    def __getitem__(self,idx):
        x,yo = self.orig_dataset.__getitem__(idx)
        yf = self.flip_label(yo)
        return x,yf

class ImageColorProj:
    def __init__(self, dataset):
        
        self.orig_dataset = dataset

    def r_proj_getitem(self, idx, dim=0):
        x,_ = self.orig_dataset.__getitem__(idx)
        r_content = self.r_proj(x,dim)
        return r_content

    def r_proj(self, x_tensor, dim=0):
        x = x_tensor[dim]
        return torch.sum(x).item()
  
