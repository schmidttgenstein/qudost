import torch
import numpy as np 
import os
import time
import sys
import json
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from qudost.density import ECDF, EPDF, RegressionCDF, DensityNetwork
from qudost.data import DataSet, DataLoader, DataGenerator
from qudost.random import density_projection
from torch.utils.data import DataLoader as TorchDataLoader


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
        s = np.random.lognormal()
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

def make_regression(x_tr,x_te,deg):
    #ECDF - Linear Regression
    epdf_eval = EPDF(x_te)
    epdf_train = EPDF(x_tr)
    reg = RegressionCDF(epdf_train.cdf, epdf_train.x_domain,degree = deg)
    x,F = epdf_train.filter_cdf(0.00001)
    _,y = epdf_train.sigma_inverse(F)
    model, poly_coeff, ypdf = reg.linear_regression(x,y)
    epdf_eval.coeff = poly_coeff
    epdf_train.coeff = poly_coeff
    return epdf_eval, epdf_train

def grad_desc_pdf_tune(epdf_eval, epdf_train, epoch= 100, lr = 0.01, lamb = 0.5):
    dn = DensityNetwork(epdf_train,epoch = epoch,lr = lr, lamb=lamb)
    ds = DataSet(epdf_train.t,epdf_train.h,tor = True,zdim = True)
    dl_tr = DataLoader(ds,batch_size = 1)
    dse = DataSet(epdf_eval.t,epdf_eval.h, tor = True,zdim = True)
    dl_eval = DataLoader(dse,batch_size = 1)
    dn.fit(dl_tr,dl_eval)
    return dn

if __name__ == "__main__":
    np.random.seed(1) # 125 is two almost separated classes
    N =  100000
    mix = 2
    ##### WITH DATAGENERATOR
    gen_dat = DataGenerator(N, n_mixture = mix, split = .5, tor = True)
    x_tr, x_te = gen_dat.gmm_data() 
    deg = 5
    epoch, lr, lamb = 100, 0.01, 0.5
    epdf_eval, epdf_train = make_regression(x_tr, x_te, deg)
    dn = grad_desc_pdf_tune(epdf_eval, epdf_train, epoch, lr, lamb)
    dn2 = grad_desc_pdf_tune(epdf_train, epdf_eval, epoch, lr, lamb)
    poly_coeff = dn2.params.detach().numpy()

    plt.figure(1)
    f_eval = dn.forward(torch.tensor(x_te,dtype = torch.float32).detach())
    plt.plot(x_te,f_eval.detach().numpy(), 'bo')
    plt.hist(x_tr,bins = 150,density = True)
    plt.title("Histogram and model")
    #plt.savefig(path_dir+"2_histogram.png")

    gmm = gau_mix(epdf_train.data, mix)
    logprob = gmm.score_samples(epdf_train.t.reshape(epdf_train.t.shape[0],1))
    gmm_pdf = np.exp(logprob)

    plt.figure(2)
    plt.plot(epdf_train.t,epdf_train.h,'.', label = 'train histo')
    plt.plot(epdf_eval.t,epdf_eval.h,'.',label = 'eval histo')
    x = torch.linspace(epdf_train.t[0],epdf_train.t[-1],490)
    f = dn.forward(torch.tensor(x,dtype = torch.float32).detach())
    f2 = dn2.forward(torch.tensor(x,dtype = torch.float32).detach())
    plt.plot(x,f2.detach().numpy(), label = 'training data model')
    plt.plot(x,f.detach().numpy(), label = 'test data model')
    plt.plot(epdf_train.t,gmm_pdf, label = 'GMM')
    pp = epdf_train.poly_eval(epdf_train.t,poly_coeff)
    plt.plot(epdf_train.t,epdf_train.sigma(pp)*(1-epdf_train.sigma(pp))*epdf_train.poly_derivative(epdf_train.t,poly_coeff), label = "LR pdf")
    plt.legend()
    plt.title("Densities")
    
    plt.figure(2)
    plt.plot(epdf_train.domain,epdf_train.cdf, label = 'Actual CDF')
    plt.plot(x,dn.net_cdf(torch.tensor(x,dtype = torch.float32)).detach().numpy(), label = 'model CDF')
    plt.plot(x, dn.activation(torch.tensor(epdf_train.poly_eval(x,epdf_train.coeff))), label = 'Linear Reg CDF')
    plt.legend()
    plt.title("CDF's")
    plt.show()
    '''
    plt.figure(6)
    # define subplot grid
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("GMM", fontsize=18, y=0.95)
    mixtures = [1,2,3,4,5,6,7,8,9]
    # loop through tickers and axes
    for mix, ax in zip(mixtures, axs.ravel()):
        gmm = gau_mix(x, mix)
        logprob = gmm.score_samples(epdf_train.t.reshape(epdf_train.t.shape[0],1))
        gmm_pdf = np.exp(logprob)
        ax.plot(epdf_train.t,epdf_train.h,'.', label = 'train histo')
        ax.plot(epdf_eval.t,epdf_eval.h,'.',label = 'eval histo')
        pp = epdf_train.poly_derivative(epdf_train.t,poly_coeff)
        sig = epdf_train.sigma(epdf_train.poly_eval(epdf_train.t,poly_coeff))
        f = sig * (1-sig) * pp
        ax.plot(epdf_train.t,f, label = 'model')
        ax.plot(epdf_train.t,gmm_pdf, label = 'GMM')
    '''
    p_value = 0.999
    ''' 
    #x_tr = torch.tensor(x_tr, dtype = torch.float32).detach()
    #x_te = torch.tensor(x_te, dtype = torch.float32).detach()
    interval_tr =dn.epdf.interval( p_value)
    te_dom_valid_pr =  dn.epdf.prob_interval(x_te,interval_tr)
    print('Train interval:', interval_tr, "L1-error + interval probability:", dn.densities_l1_distance(epdf_train.t, epdf_train.h, interval_tr))
    #print('Test interval:', interval_tr, "L1-error + interval probability:", dn.densities_l1_distance(epdf_eval.t, epdf_eval.h, interval_tr))
    parameters_data = {'data':type_data, 'number_samples':N, 'n_mixture':mix, 'degree':deg, 'learning_rate':lr, 'epochs':epoch, "lambda": lamb}
    with open(path_dir+'parameters_data.json', 'w') as file:
        json.dump(parameters_data, file, indent=4)
    plt.show() '''


    ### some of the todos: 