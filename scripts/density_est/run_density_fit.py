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

def grad_desc_pdf_tune(epdf_eval, epdf_train, epoch= 100, lr = 0.01, lamb = 0.5, sf = None):
    dn = DensityNetwork(epdf_train,epoch = epoch,lr = lr, lamb=lamb, sf=sf)
    ds = DataSet(epdf_train.t,epdf_train.h,tor = True,zdim = True)
    dl_tr = DataLoader(ds,batch_size = 1)
    dse = DataSet(epdf_eval.t,epdf_eval.h, tor = True,zdim = True)
    dl_eval = DataLoader(dse,batch_size = 1)
    dn.fit(dl_tr,dl_eval)
    return dn

if __name__ == "__main__":
    ## (Step 1) Training
    ## Seeding and fixing the size of the data and the number of mixtures, both regression and gradient descent are computed 
    np.random.seed(1) # 125 is two almost separated classes
    N =  100000
    mix = 2
    gen_dat = DataGenerator(N, n_mixture = mix, split = .5, tor = True)
    x_tr, x_te = gen_dat.gmm_data() 
    deg = 5
    epoch, lr, lamb = 100, 0.01, 0.5
    epdf_eval, epdf_train = regression(x_tr, x_te, deg)
    scale_factor = 10**(-np.round(np.log10(np.abs(epdf_train.coeff))))
    dn = grad_desc_pdf_tune(epdf_eval, epdf_train, epoch, lr, lamb, scale_factor)
    dn2 = grad_desc_pdf_tune(epdf_train, epdf_eval, epoch, lr, lamb, scale_factor)
    poly_coeff = dn2.params.detach().numpy()

    ## (Step 2) Inferences and Plots

    plt.figure(1)
    f_eval = dn.forward(torch.tensor(x_te,dtype = torch.float32).detach())
    plt.plot(x_te,f_eval.detach().numpy(), 'bo')
    plt.hist(x_tr,bins = 150,density = True)
    plt.title("Histogram and model")

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
    plt.legend()
    plt.title("Densities")
    
    plt.figure(3)
    plt.plot(epdf_train.x_domain,epdf_train.cdf, label = 'Actual CDF')
    plt.plot(x,dn.net_cdf(torch.tensor(x,dtype = torch.float32)).detach().numpy(), label = 'model CDF')
    plt.legend()
    plt.title("CDF's")
    plt.show()

    ## (Step 3) Distributional Metrics: L1-distance

    p_value = 0.99 # probability we want to estimate
    interval_tr = dn.epdf.interval(p_value) # range within the probability value is located (p_value neighborhood)
    te_dom_valid_pr =  dn.epdf.prob_interval(x_te,interval_tr) # real cdf's values within the range
    print('Train interval:', interval_tr, "L1-error + interval probability:", dn.densities_l1_distance(epdf_train.t, epdf_train.h, interval_tr)) # L1-error and probability inside the range.
    
    ## (Step 4) Distributional Metrics: Wasserstein distance
    pol1 = epdf_train.poly_eval(x,dn.params.detach().numpy())
    pol2 = epdf_eval.poly_eval(x,dn2.params.detach().numpy())
    #was = wasserstein_opt(dn.activation(pol1)[:-1],dn.activation(pol2),x)
    cdf1 = dn.activation(pol1)
    cdf2 = dn2.activation(pol2)
    was = density_projection.wasserstein_cdf(dn.activation(pol1),dn2.activation(pol2),x)
    was_scipy = stats.wasserstein_distance(x_te,x_tr)
    scipy_time = timeit.timeit(lambda: stats.wasserstein_distance(x_te,x_tr), number=1000)
    custom_time = timeit.timeit(lambda: density_projection.wasserstein_cdf(cdf1,cdf2,x), number=1000)
    print('Scipy-wasserstein distance:', was_scipy, 'scipy mean time:', scipy_time, "Fast wasserstein:", was.item(), 'fast mean time:', custom_time)
    
    #Wasserstein distance point to distribution
    wass_dis = density_projection.wasserstein_point(cdf1, 0.5, x)
    print("Wasserstein distro to point:", wass_dis.item())