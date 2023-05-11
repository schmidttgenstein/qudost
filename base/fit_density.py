
import torch 
import numpy as np 
import torch.nn as nn
from arch import MLPipeline
from data_utils import DataBatcher, DataLoader, DataSet
import torch.optim as optim
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import argrelextrema


class ECDF:
    def __init__(self,data):
        data.sort()
        self.data = data 
        self.m = data.shape[0]
        self.x_domain, self.cdf = self.ecdf()
    
    def domain(self,):
        i = 0 
        domain = []

    def ecdf_point(self,x):
        idx = self.data <= x 
        return idx.sum() / self.m
    
    def get_domain(self,eps):
        ## standin for later filtering
        return self.data
    
    def ecdf(self,):
        x = self.get_domain(eps = 0.001)
        F = np.zeros(self.m)
        for j,xj in enumerate(x):
            F[j] = self.ecdf_point(xj)
        return x,F

    def filter_cdf(self,epsilon):
        idx = (self.cdf > epsilon) & (self.cdf < 1-epsilon) 
        return self.x_domain[idx], self.cdf[idx]

    def sigma(self,t):
        return 1 / (1+np.exp(-t))

    def sigma_inverse(self,y):
        if y.min() <= 0 or y.max() >=1:
            idx = y >= 0 and y <= 1
            y = y[idx]
        return y,np.log(y/(1-y))

    def poly_eval(self,x,coeff):
        x = x.reshape([x.shape[0],1])
        x_pre = x ** np.arange(coeff.shape[0])
        return x_pre @ coeff

    def poly_derivative(self,x,coeff):
        x = x.reshape([x.shape[0],1])
        x_pre = x ** (np.arange(coeff.shape[0])-1)
        mult = coeff * np.arange(coeff.shape[0])
        return x_pre @ mult


class EPDF(ECDF):
    def __init__(self,data):
        super().__init__(data)
        self.bw = 1/((data.shape[0])**(1/3))
        n_bins = int(self.bw**(-1))
        h,t= np.histogram(data,bins = n_bins, density = True)
        t0 = t[:-1]
        t1 = t[1:]
        t = (t0+t1)/2 
        self.t = t 
        self.h = h



   

class Regression(ECDF):
    def __init__(self, ecdf_values, domain, degree):
        self.ecdf_values = ecdf_values
        self.domain = domain 
        self.degree = degree

    
    def linear_regression(self,compact, inv):
        pf = PolynomialFeatures(degree=self.degree,include_bias=False)
        lr = LinearRegression()
        ypf = pf.fit_transform(np.array(compact)[:, np.newaxis])
        lr.fit(ypf,inv)
        polynomial = np.insert(lr.coef_, 0, lr.intercept_)
        return lr, polynomial, ypf


class DensityNetwork(MLPipeline):
    def __init__(self,epdf, epoch:int= 250, lr = 0.05,):
        super().__init__(epochs = epoch, lr = lr )
        self.params = nn.Parameter(torch.tensor(epdf.coeff,dtype = torch.float32,requires_grad = True))
        self.epdf = epdf
        self.activation = nn.Sigmoid() 
        self.opt = optim.Adam([self.params],lr = lr)
        self.loss_fun = nn.MSELoss()

 
    def metrics(self,y_score,y_truth):
        with torch.no_grad():
            loss = self.loss_fun(y_score,y_truth)
        return loss
    
    def forward(self,x_in):
        p = self.poly_eval(x_in,self.params)
        p_prime = self.poly_derivative(x_in,self.params)
        sig = self.activation(p)
        f = sig * (1-sig) * p_prime
        return f
    
    def backward(self,y_score,y_truth):
        self.opt.zero_grad()
        loss = self.loss_fun(y_score,y_truth)
        loss.backward()

    def update(self,grad = None):
        self.opt.step()

    def poly_eval(self,x,coeff):
        x = x.reshape([x.shape[0],1])
        x_pre = x ** torch.arange(coeff.shape[0])
        return x_pre @ coeff

    def poly_derivative(self,x,coeff):
        x = x.reshape([x.shape[0],1])
        x_pre = x ** (torch.arange(coeff.shape[0])-1)
        mult = coeff * torch.arange(coeff.shape[0])
        return x_pre @ mult

def gen_data(n_data=10000,n_mixtures = 1,split = 0.5):
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
    return x_tr, x_te

if __name__ == "__main__":
    N =    200000
    x_tr, x_te = gen_data(N,n_mixtures = 3,split = .9)
    epdf_eval = EPDF(x_te)
    epdf_train = EPDF(x_tr)
    a_temp = argrelextrema(epdf_train.h,np.greater)
    deg = a_temp[0].shape[0]*2 + 1
    reg = Regression(epdf_eval.cdf, epdf_eval.x_domain,degree = deg)
    x,F = epdf_eval.filter_cdf(0.00001)
    _,y = epdf_eval.sigma_inverse(F)
    model, poly_coeff, ypdf = reg.linear_regression(x,y)
    epdf_eval.coeff = poly_coeff

    p = epdf_eval.poly_eval(x,poly_coeff)
    plt.figure(1)
    plt.plot(x,y)
    plt.plot(x,p)
  

    dn = DensityNetwork(epdf_eval,epoch = 250,lr = 0.0015)
    ds = DataSet(epdf_train.t,epdf_train.h,tor = True)
    dl_tr = DataLoader(ds,batch_size = 1000)
    ds = DataSet(epdf_eval.t,epdf_eval.h, tor = True)
    dl_eval = DataLoader(ds,batch_size = 500)
    dn.fit(dl_tr,dl_eval)
    f_eval = dn.forward(torch.tensor(x_te,dtype = torch.float32))
    plt.figure(3)
    plt.plot(x_te,f_eval.detach().numpy())
    plt.hist(x_tr,bins = 150,density = True)
    plt.figure(2)
    plt.plot(epdf_train.t,epdf_train.h,'.',label = 'histogram')
    pp = epdf_train.poly_derivative(epdf_train.t,poly_coeff)
    sig = epdf_train.sigma(epdf_train.poly_eval(epdf_train.t,poly_coeff))
    f = sig * (1-sig) * pp
    plt.plot(epdf_train.t,f, label = 'model')
    plt.legend()
    plt.figure(4)
    plt.plot(x,F)
    plt.plot(epdf_train.t,sig)
    plt.show()