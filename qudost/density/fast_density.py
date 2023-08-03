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
    
    def ecdf_point(self,x,data=None):
        if data is None:
            data = self.data 
        m = data.shape[0]
        idx = data <= x 
        return idx.sum() / m

    def get_domain(self,eps):
        ## standin for later filtering
        return self.data
    
    def ecdf(self,x = None):
        if x is None:
            x = self.get_domain(eps = 0.001)
        else:
            x.sort()
        m = x.shape[0]
        F = np.zeros(m)
        for j,xj in enumerate(x):
            F[j] = (j+1)/m #self.ecdf_point(xj,x)
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

    def interval(self, p_value):
        ### self.data x_arg which realizes p_value with ecdf_point
        #sorted_x = torch.sort(x_in)[0]
        idx_left = np.argmin(np.abs((1-p_value)/2 - self.cdf))
        x_left = self.data[idx_left]
        idx_right = np.argmin(np.abs(p_value +(1-p_value)/2 - self.cdf))
        x_right = self.data[idx_right]
        #index = int(p_value*x_in.size(0))
        #x_left = sorted_x[0]
        #x_right = sorted_x[index]
        return x_left, x_right
    
    def prob_interval(self, x_in, interval):
        #total_points = x_in.size(0)
        #points_within = torch.sum((x_in >= interval[0]) & (x_in <= interval[1]))
        F_left = self.ecdf_point(interval[0],x_in)
        F_right = self.ecdf_point(interval[1],x_in)
        return  F_right - F_left #points_within/total_points

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
    def __init__(self,epdf, epoch:int= 250, lr = 0.01, lamb=0.5):
        super().__init__(epochs = epoch, lr = lr )
        self.params = nn.Parameter(torch.tensor(epdf.coeff,dtype = torch.float32,requires_grad = True))
        self.lamb = lamb
        self.epdf = epdf
        self.activation = nn.Sigmoid() 
        self.opt = optim.Adam([self.params],lr = lr)
        self.loss_fun = nn.MSELoss()
        #self.loss_fun = nn.L1Loss()
        #self.loss_fun = nn.KLDivLoss()
    
    def mod_loss(self, y_score, y_truth, x_in):
        ### CLEAN ME PLEASE :D 
        p = self.poly_eval(x_in,self.params)
        sig = self.activation(p)
        epdf_cdf = 0 * sig
        for j,x in enumerate(x_in):
            epdf_cdf[j] = self.epdf.ecdf_point(x) 
        
        #loss = self.lamb*((1+y_truth)*(y_score-y_truth)**2).mean() + (1-self.lamb)*((sig-epdf_cdf)**2).mean() 
        loss = self.lamb*((y_score-y_truth)**2).mean() + (1-self.lamb)*((sig-epdf_cdf)**2).mean() 
        return loss
    

    def metrics(self,y_score,y_truth):
        with torch.no_grad():
            loss = self.loss_fun(y_score,y_truth)
        return loss
    
    def forward(self,x_in):
        ### TOCA MANDAR ACA UN SIGMA INVERSE??
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
    
    def net_cdf(self,x_in):
        p = self.poly_eval(x_in,self.params)
        sig = self.activation(p)
        return sig

    def densities_l1_distance(self, x_in, h_values, interval):
        ## Input x_values, h_values, left-right trim interval.
        ## Output: L1-error and probability inside the range.
        ##
        model_values = self.forward(torch.tensor(x_in, dtype = torch.float32).detach())
        l1_dif_sum = ((torch.abs(h_values - model_values) * (x_in.diff().mean())).sum()).item()
        return l1_dif_sum, self.epdf.prob_interval(x_in, interval)