import torch
import numpy as np 
import os
import time
import sys
import json
import wandb
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from qudost.density import ECDF, EPDF, RegressionCDF, DensityNetwork
from qudost.data import DataSet, DataLoader, DataGenerator
from torch.utils.data import DataLoader as TorchDataLoader

wandb.login()

def fat_tail(N = 10000, split=0.5, tor = False):
    y = np.random.random(N)
    u = np.zeros(y.shape[0])
    idx1 =  y<=1/4
    u[idx1] =  -1/(4*y[idx1])
    idx2 = (y>1/4) & (y<=3/4)
    u[idx2] = 4*y[idx2]-2
    idx3 = y>3/4
    u[idx3] = 1/(4*(1-y[idx3]))
    u = u[(u>-20)&(u<20)]
    np.random.shuffle(u)
    split_idx = int(split * u.shape[0])
    x_tr = u[:split_idx]
    x_te = u[split_idx:]
    if tor: 
        x_tr = torch.tensor(x_tr)
        x_te = torch.tensor(x_te)
    return x_tr, x_te

def gen_data(n_data=10000,n_mixtures = 1,split = 0.5,tor = False):
    datas = []
    for j in range(n_mixtures):
        m = np.random.normal(0,5)
        s = 2*np.random.lognormal()
        data = np.random.normal(m,s,n_data)
        datas.append(data)
    data = np.concatenate(datas)
    np.random.shuffle(data)
    split_idx = int(split * data.shape[0])
    x_tr = data[:split_idx]
    x_te = data[split_idx:]
    if tor: 
        x_tr = torch.tensor(x_tr)
        x_te = torch.tensor(x_te)
    return x_tr, x_te

def gen_cauchy(n_data=10000, split=0.5, tor = False):
    s = np.random.standard_cauchy(n_data)
    s = s[(s>-20) & (s<20)]
    np.random.shuffle(s)
    split_idx = int(split * s.shape[0])
    x_tr = s[:split_idx]
    x_te = s[split_idx:]
    if tor: 
        x_tr = torch.tensor(x_tr)
        x_te = torch.tensor(x_te)
    return x_tr, x_te


def sigma(t,a=1,b=0,):
        s = 1/(1+np.exp(-(t-b)/a))
        return s 

def simulate_multimodal_cdf(n_modes, range_dom, resolution=10000):
    cdf = np.zeros(resolution)
    pdf = np.zeros(resolution)
    domain = np.linspace(min(range_dom) - 3*np.std(range_dom), max(range_dom) + 3*np.std(range_dom), resolution)
    for i in range(n_modes):
         a = np.random.lognormal()
         b = np.random.randint(-10,10)
         sig = sigma(domain,a,b)
         cdf = np.add(cdf,sig)
         pdf = np.add(pdf,1/a*(sig)*(1-sig))
    cdf = 1/n_modes*(cdf)
    pdf = 1/n_modes*(pdf)
    return domain,cdf,pdf

def inverse_transform_sampling(domain, cdf, num_samples, split):
    icdf = np.interp(np.random.uniform(0, 1, num_samples), cdf, domain)
    np.random.shuffle(icdf)
    split_idx = int(split * icdf.shape[0])
    x_tr = icdf[:split_idx]
    x_te = icdf[split_idx:]
    return x_tr, x_te
    #return icdf

def gau_mix(fit_samples,n_mixtures = 1):
    gmm = GaussianMixture(n_mixtures)
    fit_samples=fit_samples.reshape(-1,1)
    gmm.fit(fit_samples)
    return gmm

if __name__ == "__main__":
    wandb.login()
    np.random.seed(1) # 125 is two almost separated classes
    fname = str(time.time())
    N =  int(10**5)
    mix = 3
    gen_dat = DataGenerator(N, n_mixture = mix, split = .5, tor = True)
    x_tr, x_te = gen_dat.gmm_data()
    type_data = "gaussian mixture" 

    epdf_eval = EPDF(x_te)
    epdf_train = EPDF(x_tr)
    a_temp = argrelextrema(epdf_train.h.detach().numpy(),np.greater)
    deg = 5
    reg = RegressionCDF(epdf_train.cdf, epdf_train.x_domain,degree = deg)
    x,F = epdf_train.filter_cdf(0.00001)
    _,y = epdf_train.sigma_inverse(F)
    model, poly_coeff, ypdf = reg.linear_regression(x,y)
    epdf_eval.coeff = poly_coeff
    epdf_train.coeff = poly_coeff
    scale_factor = 10**(-np.round(np.log10(np.abs(poly_coeff))))

    p = epdf_train.poly_eval(x,poly_coeff)
  
    ''' 
    plt.figure(1)
    plt.plot(x,y, label = 'sigma inverse cdf')
    plt.plot(x,p, label = 'polynomial')
    plt.title("Sigma inverse regression")
    plt.legend() '''

    #WANDB

    epoch, lr, lamb = 500, 0.005, .5
    sweep_config = {'method':'random'}
    metric = {'name': 'loss', 'goal': 'minimize'}
    parameters_dict = {'lambda': {'values': [0.3, 0.5, 0.7]}, 'learning_rate': {'distribution': 'uniform','max': 0.1,'min': 0}, 'epochs': [100, 200, 500, 1000]}
    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="cpdf")
    run = wandb.init(project="cpdf", config={"learning_rate": lr, "epochs": epoch, "lambda": lamb})

    dn = DensityNetwork(epdf_train,epoch = epoch,lr = lr, lamb=lamb,sf = scale_factor)
    ds = DataSet(epdf_train.t,epdf_train.h,tor = True,zdim = True)
    dl_tr = DataLoader(ds,batch_size = 100)
    dse = DataSet(epdf_eval.t,epdf_eval.h, tor = True,zdim = True)
    dl_eval = DataLoader(dse,batch_size = 50)
    
    orig_stdout = sys.stdout
    
    dn.fit(dl_tr,dl_eval)


    #f_eval = dn.forward(x_te.clone())
    if isinstance(x_te,torch.Tensor):
        x_te = x_te.clone().detach().float()
    else:
        x_te = torch.tensor(x_te,dtype = torch.float32)
    '''
    plt.figure(2)
    f_eval = dn.forward(x_te)
    plt.plot(x_te,f_eval.clone().detach().numpy(), 'bo')
    plt.hist(x_tr,bins = 150,density = True)
    plt.title("Histogram and model")
    #plt.savefig(path_dir+"2_histogram.png") '''

    gmm = gau_mix(x, mix)
    logprob = gmm.score_samples(epdf_train.t.reshape(epdf_train.t.shape[0],1))
    gmm_pdf = np.exp(logprob)

    ''' 
    plt.figure(3)
    plt.plot(epdf_train.t,epdf_train.h,'.', label = 'train histo')
    plt.plot(epdf_eval.t,epdf_eval.h,'.',label = 'eval histo') '''
    ####
    if isinstance(epdf_train.t,torch.Tensor):
        epdft = epdf_train.t.clone().detach().float()
    else:
        epdft = torch.tensor(epdf_train.t,dtype = torch.float32)
    f2 = dn.forward(epdft)
    plt.plot(epdf_train.t,gmm_pdf, label = 'GMM')
    plt.plot(epdf_train.t,f2.detach(),label = 'actual model')
    pp = epdf_eval.poly_eval(epdf_train.t,poly_coeff)
    plt.plot(epdf_train.t,epdf_train.sigma(pp)*(1-epdf_train.sigma(pp))*epdf_train.poly_derivative(epdf_train.t,poly_coeff), label = "LR pdf")
    #plt.plot(domain, pdf, label = 'true pdf')
    plt.legend()
    plt.title("Densities")
    #plt.savefig(path_dir+"3_densities.png")
    
    plt.figure(4)
    plt.plot(x,F, label = 'Actual CDF')
    plt.plot(x,dn.net_cdf(torch.tensor(x,dtype = torch.float32)).detach().numpy(), label = 'model CDF')
    plt.plot(x, dn.activation(torch.tensor(p)), label = 'Linear Reg CDF')
    plt.legend()
    plt.title("CDF's")
    plt.show()
   