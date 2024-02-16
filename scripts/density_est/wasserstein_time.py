import torch
import numpy as np
import timeit
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
from run_density_fit import grad_desc_pdf_tune, regression

if __name__ == "__main__":
    # (Step 1) Generate data 1 and train a density network
    np.random.seed(125) 
    # 125 BD1 
    # 2 is like a single class we do slightly better than gmm 
    # 123 is a single with gmm better
    # 12 is UD1
    # 20 is BD2
    # FT1 is gen_dat.cauchy(trim = 20) seed 20
    N =  20000
    mix = 2
    gen_dat = DataGenerator(N, n_mixture = mix, split = .5, tor = True)
    #x_tr, x_te = gen_dat.cauchy(trim = 20)
    x_tr, x_te = gen_dat.gmm_data() 
    x1 = torch.concatenate((x_tr,x_te))
    deg = 5
    epoch, lr, lamb = 100, 0.0001, 0.4
    epdf_eval, epdf_train = regression(x_tr, x_te, deg)
    scale_factor = 10**(-np.round(np.log10(np.abs(epdf_train.coeff))))
    dn = grad_desc_pdf_tune(epdf_train, epdf_eval, epoch, lr, lamb, scale_factor)

    # (Step 2) Generate data 2 and train a density network
    
    np.random.seed(12)
    gen_dat = DataGenerator(N, n_mixture = mix, split = .5, tor = True)
    x_tr2, x_te2 = gen_dat.gmm_data()
    x2 = torch.concatenate((x_tr2,x_te2))
    epdf_eval2, epdf_train2 = regression(x_tr2, x_te2, deg)
    dn2 = grad_desc_pdf_tune(epdf_train2, epdf_eval2, epoch, lr, lamb, scale_factor)

    ## (Step 3) Wasserstein distance computational time
    #Domain
    x = torch.linspace(min(torch.min(x1),torch.min(x2)),max(torch.max(x1),torch.max(x2)),N)
    #CDFs
    cdf1 = dn.net_cdf(x).detach()
    cdf2 = dn2.net_cdf(x).detach()
    #PDFs
    pdf1 = dn.net_pdf(x).detach()
    pdf2 = dn2.net_pdf(x).detach()

    was = density_projection.wasserstein_cdf(cdf1,cdf2,x)
    was_scipy = stats.wasserstein_distance(x1,x2)
    #time
    num_runs = 35000
    scipy_time = timeit.repeat(lambda: stats.wasserstein_distance(x1,x2), number=1, repeat=num_runs)
    custom_time = timeit.repeat(lambda: density_projection.wasserstein_cdf(cdf1,cdf2,x), number=1, repeat=num_runs)
    print('Scipy-wasserstein distance: ', was_scipy, ' mean time: ', np.mean(scipy_time), ' std: ', np.std(scipy_time))
    print("Fast wasserstein distance: ", was.item(), ' mean time: ', np.mean(custom_time), ' std: ', np.std(custom_time))
