import torch
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from qudost import ECDF, EPDF, RegressionCDF, DensityNetwork
from qudost import DataSet, DataLoader


def gen_data(n_data=10000,n_mixtures = 1,split = 0.5):
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
    return x_tr, x_te


if __name__ == "__main__":
    N =    50000
    x_tr, x_te = gen_data(N,n_mixtures = 3,split = .5)
    epdf_eval = EPDF(x_te)
    epdf_train = EPDF(x_tr)
    a_temp = argrelextrema(epdf_train.h,np.greater)
    deg = a_temp[0].shape[0]*2 + 3
    reg = RegressionCDF(epdf_eval.cdf, epdf_eval.x_domain,degree = deg)
    x,F = epdf_eval.filter_cdf(0.00001)
    _,y = epdf_eval.sigma_inverse(F)
    model, poly_coeff, ypdf = reg.linear_regression(x,y)
    epdf_eval.coeff = poly_coeff

    p = epdf_eval.poly_eval(x,poly_coeff)
    plt.figure(1)
    plt.plot(x,y)
    plt.plot(x,p)
  
    dn = DensityNetwork(epdf_eval,epoch = 250,lr = 0.00075)
    ds = DataSet(epdf_train.t,epdf_train.h,tor = True,zdim = True)
    dl_tr = DataLoader(ds,batch_size = 1000)
    ds = DataSet(epdf_eval.t,epdf_eval.h, tor = True,zdim = True)
    dl_eval = DataLoader(ds,batch_size = 500)
    dn.fit(dl_tr,dl_eval)

    plt.figure(2)
    f_eval = dn.forward(torch.tensor(x_te,dtype = torch.float32))
    plt.plot(x_te,f_eval.detach().numpy())
    plt.hist(x_tr,bins = 150,density = True)

    plt.figure(3)
    plt.plot(epdf_train.t,epdf_train.h,'.',label = 'train histo')
    plt.plot(epdf_eval.t,epdf_eval.h,'.',label = 'eval histo')
    pp = epdf_train.poly_derivative(epdf_train.t,poly_coeff)
    sig = epdf_train.sigma(epdf_train.poly_eval(epdf_train.t,poly_coeff))
    f = sig * (1-sig) * pp
    plt.plot(epdf_train.t,f, label = 'model')
    plt.legend()
    plt.figure(4)
    plt.plot(x,F)
    plt.plot(epdf_train.t,sig)
    plt.show()