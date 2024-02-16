import torch
import numpy as np 
import os
import timeit
import sys
import json
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import optimize, stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from qudost.density import ECDF, EPDF, RegressionCDF, DensityNetwork
from qudost.data import DataSet, DataLoader, DataGenerator
from qudost.random import density_projection
from torch.utils.data import DataLoader as TorchDataLoader


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

def regression(x_tr,x_te,deg):
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

def grad_desc_pdf_tune(epdf_train, epdf_eval, epoch= 100, lr = 0.01, lamb = 0.5, sf = None):
    dn = DensityNetwork(epdf_train,epoch = epoch,lr = lr, lamb=lamb, sf=sf)
    ds = DataSet(epdf_train.t,epdf_train.h,tor = True,zdim = True)
    dl_tr = DataLoader(ds,batch_size = 1)
    dse = DataSet(epdf_eval.t,epdf_eval.h, tor = True,zdim = True)
    dl_eval = DataLoader(dse,batch_size = 1)
    dn.fit(dl_tr,dl_eval)
    return dn

if __name__ == "__main__":
    ## (Step 1) Data generation
    ## Seeding and fixing the size of the data and the number of mixtures
    np.random.seed(2) 
    # 125 BD1 
    # 2 is like a single class we do slightly better than gmm 
    # 123 is a single with gmm better
    # 12 is UD1
    # 20 is BD2
    # FT1 is gen_dat.cauchy(trim = 20) seed 20
    N =  100000
    mix = 2
    gen_dat = DataGenerator(N, n_mixture = mix, split = .5, tor = True)
    #x_tr, x_te = gen_dat.cauchy(trim = 20)
    x_tr, x_te = gen_dat.gmm_data()
    mixtures = gen_dat.gmm_mixtures

    ## (Step 2) Training
    # both regression and gradient descent
    deg = 5
    epoch, lr, lamb = 100, 0.0001, 0.4
    epdf_eval, epdf_train = regression(x_tr, x_te, deg)
    scale_factor = 10**(-np.round(np.log10(np.abs(epdf_train.coeff))))
    #dn2 = grad_desc_pdf_tune(epdf_eval, epdf_train, epoch, lr, lamb, scale_factor)
    dn = grad_desc_pdf_tune(epdf_train, epdf_eval, epoch, lr, lamb, scale_factor)
    poly_coeff = dn.params.detach().numpy()

    ## (Step 3) Inferences and Plots

    plt.figure(1)
    f_eval = dn.net_pdf(torch.tensor(x_te,dtype = torch.float32).detach())
    plt.plot(x_te,f_eval.detach().numpy(), '*', color = 'blue', label = 'Model values on test data')
    plt.hist(x_tr,bins = 150,density = True, alpha=.7, label='Histogram')
    plt.legend()
    plt.title("Histogram and PDF model")

    gmm = gau_mix(epdf_train.data, mix)
    logprob = gmm.score_samples(epdf_train.t.reshape(epdf_train.t.shape[0],1))
    gmm_pdf = np.exp(logprob)

    plt.figure(2)
    plt.plot(epdf_train.t,epdf_train.h,'.', label = 'Train histo.')
    plt.plot(epdf_eval.t,epdf_eval.h,'.',label = 'Eval. histo.')
    x = torch.linspace(epdf_train.t[0],epdf_train.t[-1],490)
    f = dn.net_pdf(torch.tensor(x,dtype = torch.float32).detach())
    plt.plot(x,f.detach().numpy(), label = 'PDF model')
    plt.plot(epdf_train.t,gmm_pdf,'-.' ,label = 'GMM')
    plt.legend(loc='upper right')
    plt.title("Densities")
    
    plt.figure(3)
    plt.plot(epdf_train.x_domain,epdf_train.cdf, label = 'Empirical ("true") CDF')
    plt.plot(x,dn.net_cdf(torch.tensor(x,dtype = torch.float32)).detach().numpy(), label = 'Model CDF')
    plt.legend()
    plt.title("CDF's")
    plt.show()