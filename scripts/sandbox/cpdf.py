import torch
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from qudost.density import ECDF, EPDF, RegressionCDF, DensityNetwork
from qudost.data import DataSet, DataLoader
from torch.utils.data import DataLoader as TorchDataLoader


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
    N =  100000
    mix = 2
    x_tr, x_te = gen_data(N,n_mixtures = mix,split = .9,tor= True)

    n_modes, range_dom = mix, [-5,5]
    #domain, cdf, pdf = simulate_multimodal_cdf(n_modes, range_dom, resolution=N)
    #x_tr, x_te = inverse_transform_sampling(domain, cdf, num_samples=N, split = .5)
    epdf_eval = EPDF(x_te)
    epdf_train = EPDF(x_tr)
    a_temp = argrelextrema(epdf_train.h.detach().numpy(),np.greater)
    deg = a_temp[0].shape[0]*2 + 3
    reg = RegressionCDF(epdf_eval.cdf, epdf_eval.x_domain,degree = deg)
    x,F = epdf_eval.filter_cdf(0.00001)
    _,y = epdf_eval.sigma_inverse(F)
    model, poly_coeff, ypdf = reg.linear_regression(x,y)
    epdf_eval.coeff = poly_coeff
    epdf_train.coeff = poly_coeff

    p = epdf_eval.poly_eval(x,poly_coeff)
    plt.figure(1)
    plt.plot(x,y)
    plt.plot(x,p)

    '''
    plt.figure(5)
    plt.plot(domain, pdf, label = "true pdf")
    plt.plot(x,epdf_train.sigma(p)*(1-epdf_train.sigma(p))*epdf_train.poly_derivative(x,poly_coeff), label = "LR pdf")
    plt.legend() '''
  
    dn = DensityNetwork(epdf_train,epoch = 500,lr = 0.01, lamb=0.5)
    ds = DataSet(epdf_train.t,epdf_train.h,tor = True,zdim = True)
    dl_tr = DataLoader(ds,batch_size = 1000)
    dse = DataSet(epdf_eval.t,epdf_eval.h, tor = True,zdim = True)
    dl_eval = DataLoader(dse,batch_size = 500)
    dn.fit(dl_tr,dl_eval)

    plt.figure(2)
    f_eval = dn.forward(torch.tensor(x_te,dtype = torch.float32))
    plt.plot(x_te,f_eval.detach().numpy())
    plt.hist(x_tr,bins = 150,density = True)

    gmm = gau_mix(x, mix)
    logprob = gmm.score_samples(epdf_train.t.reshape(epdf_train.t.shape[0],1))
    gmm_pdf = np.exp(logprob)
    plt.figure(3)
    plt.plot(epdf_train.t,epdf_train.h,'.', label = 'train histo')
    plt.plot(epdf_eval.t,epdf_eval.h,'.',label = 'eval histo')
    pp = epdf_train.poly_derivative(epdf_train.t,poly_coeff)
    sig = epdf_train.sigma(epdf_train.poly_eval(epdf_train.t,poly_coeff))
    f = sig * (1-sig) * pp
    plt.plot(epdf_train.t,f, label = 'model')
    plt.plot(epdf_train.t,gmm_pdf, label = 'GMM')
    #plt.plot(domain, pdf, label = 'true pdf')

    plt.legend()
    plt.figure(4)
    plt.plot(x,F)
    plt.plot(epdf_train.t,sig)
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
    plt.show()