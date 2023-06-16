import torch 
import numpy as np 
import torch.nn as nn
from qudost import MLPipeline
import torch.optim as optim
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import argrelextrema


class ECDF:
    def __init__(self,data):
        data, _  = data.sort()
        self.data = data 
        self.m = data.shape[0]
        self.x_domain, self.cdf = self.ecdf()
    
    def domain(self,):
        i = 0 
        domain = []
    
    def ecdf_point(self,x):
        idx = self.data <= x 
        return idx.sum() / self.m
    '''
    def ecdf_point(self, x):
        idx = self.data <= x
        return idx.sum(axis=0) / self.m
    '''
    def get_domain(self,eps):
        ## standin for later filtering
        return self.data
    
    def ecdf(self,):
        x = self.get_domain(eps = 0.001)
        F = np.zeros(self.m)
        for j,xj in enumerate(x):
            F[j] = self.ecdf_point(xj)
        return x,F
    
    '''
    def ecdf(self,):
        x = self.get_domain(eps=0.001)
        idx = self.data[:, np.newaxis] <= x
        F = idx.sum(axis=0) / self.m
        return x, F
    '''
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
        h,t= torch.histogram(data,bins = n_bins, density = True)
        t0 = t[:-1]
        t1 = t[1:]
        t = (t0+t1)/2 
        self.t = t 
        self.h = h
   

class RegressionCDF(ECDF):
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
    def __init__(self,epdf, epoch:int= 250, lr = 0.05, lamb=0.5):
        super().__init__(epochs = epoch, lr = lr )
        self.params = nn.Parameter(torch.tensor(epdf.coeff,dtype = torch.float32,requires_grad = True))
        self.lamb = lamb
        self.epdf = epdf
        self.activation = nn.Sigmoid() 
        self.opt = optim.Adam([self.params],lr = lr)
        self.loss_fun = nn.MSELoss()

    
    def mod_loss(self, y_score, y_truth, x_in): ###Error: x_in is being read as a numpy array, don't know why if comes from a DataLoader
        ### CLEAN ME PLEASE :D 
        #pre_loss = self.loss_fun(y_score,y_truth)
        p = self.poly_eval(x_in,self.params)
        sig = self.activation(p)
        ### Loss: lamda(sigma(p)(1-sigma(p))p'-histogram) + (1-lambda)(sigma(p)-empirical cdf)
        ### Pin here, need to line up sizing between sig + epdf."cdf"
        epdf_cdf = 0 * sig
        ## fix this implementation 
        for j,x in enumerate(x_in):
            epdf_cdf[j] = self.epdf.ecdf_point(x)
        #mse_weighted = self.lamb*((1+y_truth)*(y_score-y_truth)**2)
        #loss = mse_weighted.mean() + (1-self.lamb)*((sig-epdf_cdf)**2).mean() 
        loss = self.lamb*((y_score-y_truth)**2).mean() + (1-self.lamb)*((sig-epdf_cdf)**2).mean() 
        return loss
    
    '''
    def mod_loss(self, y_score, y_truth, x_in):
        p = self.poly_eval(x_in, self.params)
        sig = self.activation(p)
        epdf_cdf = self.epdf.ecdf_point(x_in)
        loss = self.lamb * ((y_score - y_truth) ** 2).mean() + (1 - self.lamb) * ((sig - epdf_cdf) ** 2).mean()
        return loss
    '''

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
    
    def backward(self,y_score,y_truth, x_in):
        self.opt.zero_grad()
        loss = self.mod_loss(y_score,y_truth, x_in)
        #loss = self.loss_fun(y_score,y_truth)
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
