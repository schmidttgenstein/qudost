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
        ### Initializes the ECDF class instance. Sorts the input data, computes the ECDF, and stores the data and the ECDF.
        ### Input: data (Tensor) - The input data for which the empirical cumulative distribution function (ECDF) will be computed.
        data, _  = data.sort()
        self.data = data 
        self.m = data.shape[0]
        self.x_domain, self.cdf = self.ecdf(data)
    

    def ecdf_point(self,x,data=None):
        ### Computes the value of the ECDF at a given point x.
        ### Input: 
        ###   x (float) - The point at which to evaluate the ECDF.
        ###   data (Tensor, optional) - The data used to compute the ECDF. If not provided, uses the stored data in the class instance.
        ### Output: ECDF value at point x.
        if data is None:
            data = self.data 
        m = data.shape[0]
        idx = data <= x 
        return idx.sum() / m

    def get_domain(self,eps):
        ### Retrieves the domain of the ECDF. (Standing for later filtering.)
        ### Input: eps (float) - Tolerance parameter for filtering.
        ### Output: Domain of the ECDF.
        return self.data
    
    def ecdf(self,x = None, alpha = 1000):
        ### Computes the empirical cumulative distribution function (ECDF) for the given domain.
        ### Input: 
        ###   x (Tensor, optional) - The domain over which to compute the ECDF. If not provided, uses the domain obtained from get_domain().
        ###   alpha (int, default: 1000) - Parameter for interpolation.
        ### Output: Tuple containing the sorted domain x and the corresponding ECDF F.
        if x is None:
            x = self.get_domain(eps = 0.001)
        x.sort()
        m = x.shape[0]
        F = np.linspace(0,1,m)
        ''' want interpolation for ecdf? 
        for j,xj in enumerate(x):
            F[j] = (j+1)/m 
        x_interpolated, F_interpolated = self.fill_gaps(x, F, alpha)
        '''
        return x,F
    
    def fill_gaps(self, x, F, alpha):
        ### Interpolates gaps in the ECDF domain and values using linear interpolation.
        ### Input: 
        ###   x (Tensor) - Sorted domain values.
        ###   F (Tensor) - ECDF values corresponding to the domain.
        ###   alpha (int) - Parameter for interpolation.
        ### Output: Tuple containing the interpolated domain and corresponding ECDF values.
        x_interpolated = [x[0]]
        F_interpolated = [F[0]]

        for i in range(1, len(x)):
            if x[i] - x[i - 1] > 1e-1:
                num_points = int(alpha)
                x_interp = np.linspace(x[i - 1], x[i], num_points + 2)[1:-1]
                F_interp = np.interp(x_interp, [x[i - 1], x[i]], [F[i - 1], F[i]])
                x_interpolated.extend(x_interp)
                F_interpolated.extend(F_interp)

            x_interpolated.append(x[i])
            F_interpolated.append(F[i])

        return np.array(x_interpolated), np.array(F_interpolated)
    
    def filter_cdf(self,epsilon):
        ### Filters the ECDF by removing values close to 0 and 1 based on a tolerance parameter.
        ### Input: epsilon (float) - Tolerance parameter for filtering.
        ### Output: Filtered domain and ECDF values.
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
        ### Evaluates a polynomial function at the given input values.
        ### Input: 
        ###   x (Tensor) - Input values.
        ###   coeff (Tensor) - Polynomial coefficients.
        ### Output: Evaluated polynomial values.
        x = x.reshape([x.shape[0],1])
        x_pre = x ** np.arange(coeff.shape[0])
        return x_pre @ coeff

    def poly_derivative(self,x,coeff):
        ### Evaluates the derivative of a polynomial function at the given input values.
        ### Input:
        ###   x (Tensor) - Input values.
        ###   coeff (Tensor) - Polynomial coefficients.
        x = x.reshape([x.shape[0],1])
        x_pre = x ** (np.arange(coeff.shape[0])-1)
        mult = coeff * np.arange(coeff.shape[0])
        return x_pre @ mult

    def interval(self, p_value):
        ### Computes the interval corresponding to a given probability value.
        ### Input: p_value (float) - Probability value.
        ### Output: Tuple containing the interval corresponding to the given probability.
        idx_left = np.argmin(np.abs((1-p_value)/2 - self.cdf))
        x_left = self.x_domain[idx_left]
        idx_right = np.argmin(np.abs(p_value +(1-p_value)/2 - self.cdf))
        x_right = self.x_domain[idx_right]
        return x_left, x_right
    
    def prob_interval(self, x_in, interval):
        ### Computes the probability of input values falling within a given interval.
        ### Input:
        ###   x_in (Tensor) - Input values.
        ###   interval (Tuple) - Interval for which to compute the probability.
        F_left = self.ecdf_point(interval[0],x_in)
        F_right = self.ecdf_point(interval[1],x_in)
        return  F_right - F_left 

class EPDF(ECDF):
    ### Initializes the EPDF class instance. Inherits from ECDF class. Computes additional histogram parameters for later PDF fine-tune.
    ### Input: data (Tensor) - The input data for which the empirical cumulative distribution function (ECDF) will be computed.
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
        ### Initializes the RegressionCDF class instance with ECDF values, domain, and polynomial degree.
        ### Input: 
        ###   ecdf_values (array-like) - ECDF values.
        ###   domain (array-like) - Domain values corresponding to the ECDF values.
        ###   degree (int) - Degree of polynomial regression.
        self.ecdf_values = ecdf_values
        self.domain = domain 
        self.degree = degree

    
    def linear_regression(self,compact, inv):
        ### Performs linear regression to fit a polynomial model to the sigmoid inverse of data.
        ### Input: 
        ###   compact (array-like) - input values filtered.
        ###   inv (array-like) - Sigmoid inverse values corresponding to the compact input.
        ### Output: 
        ###   lr (LinearRegression object) - Fitted linear regression model.
        ###   polynomial (array-like) - Coefficients of the fitted polynomial.
        ###   ypf (array-like) - Transformed input values.
        pf = PolynomialFeatures(degree=self.degree,include_bias=False)
        lr = LinearRegression()
        ypf = pf.fit_transform(np.array(compact)[:, np.newaxis])
        lr.fit(ypf,inv)
        polynomial = np.insert(lr.coef_, 0, lr.intercept_)
        return lr, polynomial, ypf


class DensityNetwork(MLPipeline):
    def __init__(self,epdf, epoch:int= 250, lr = 0.01, lamb=0.5,sf = None, pot_weight = 1000):
        ### Initializes the DensityNetwork class instance with parameters and initializes optimization.
        ### Input:
        ###   epdf (EPDF object) - Instance of the EPDF class containing ECDF values and coefficients.
        ###   epoch (int, default: 250) - Number of training epochs.
        ###   lr (float, default: 0.01) - Learning rate for optimization.
        ###   lamb (float, default: 0.5) - Lambda parameter for loss function.
        ###   sf (float or None, optional) - Scaling factor for coefficients. If None, uses 1.
        ###   pot_weight (float, default: 1000) - Potential weight parameter for partition unity.
        super().__init__(epochs = epoch, lr = lr )
        params = torch.tensor(epdf.coeff,dtype = torch.float32) #,requires_grad = True)
        self.pot_weight = pot_weight
        if sf is None:
            self.sf = torch.tensor(0*epdf.coeff + 1,dtype = torch.float32)
        else:
            self.sf = torch.tensor(sf,dtype = torch.float32) 
            params = self.sf * params 
        self.params = nn.Parameter(params,requires_grad = True)
        self.lamb = lamb
        self.epdf = epdf
        self.activation = nn.Sigmoid() 
        self.opt = optim.Adam([self.params],lr = lr)
        self.loss_fun = nn.MSELoss()

    def neg_pen(self, y_score, a=1e2, b=2):
        ### Applies negative penalty to predicted values less than 0.
        ### Input:
        ###   y_score (Tensor) - Predicted values.
        ###   a (float, default: 1e2) - Parameter a for negative penalty.
        ###   b (float, default: 2) - Parameter b for negative penalty.
        ### Output: Tensor with negative penalty applied.
        return torch.where(y_score < 0, b*torch.exp(-1/a*(y_score)), torch.zeros_like(y_score))
    
    def mod_loss(self, y_score, y_truth, x_in):
        ### Computes the modified loss function including regularization terms.
        ### Input:
        ###   y_score (Tensor) - Predicted values.
        ###   y_truth (Tensor) - Ground truth (histogram) values.
        ###   x_in (Tensor) - Input values.
        coeffs = self.params / self.sf 
        p = self.poly_eval(x_in,coeffs)
        sig = self.activation(p)
        epdf_cdf = 0 * sig
        for j,x in enumerate(x_in):
            epdf_cdf[j] = self.epdf.ecdf_point(x) 
        loss = self.lamb*((y_score-y_truth)**2).mean() + (1-self.lamb)*((sig-epdf_cdf)**2).mean() # + self.neg_pen(y_score).mean()
        return loss
    

    def metrics(self,y_score,y_truth):
        ### Computes the MSE loss function value.
        ### Input:
        ###  y_score (Tensor) - Predicted values.
        ###  y_truth (Tensor) - Ground truth (histogram) values.
        with torch.no_grad():
            loss = self.loss_fun(y_score,y_truth)
        return loss
    
    def forward(self,x_in):
        ### Forward pass of the density network.
        ### Input: x_in (Tensor) - Input values.
        ### Output: Predicted values.
        coeffs = self.params / self.sf 
        p = self.poly_eval(x_in,coeffs)
        p_prime = self.poly_derivative(x_in,coeffs)
        sig = self.activation(p)
        f = sig * (1-sig) * p_prime
        return f
    
    def backward(self,y_score,y_truth, x_in):
        ### Backward pass of the density network.
        ### Input:
        ###   y_score (Tensor) - Predicted values.
        ###   y_truth (Tensor) - Ground truth (histogram) values.
        self.opt.zero_grad()
        loss = self.mod_loss(y_score,y_truth, x_in)
        loss.backward()

    def update(self,grad = None):
        ### Updates the parameters of the density network.
        ### Input: grad (Tensor, optional) - Gradient values.
        self.opt.step()

    def poly_eval(self,x,coeff):
        ### Evaluates a polynomial with coefficients coeff at input values x.
        ### Input:
        ###   x (Tensor) - Input values.
        ###   coeff (Tensor) - Polynomial coefficients.
        ### Output: Polynomial evaluated at input x.
        x = x.reshape([x.shape[0],1])
        x_pre = x ** torch.arange(coeff.shape[0])
        return x_pre @ coeff

    def poly_derivative(self,x,coeff):
        ### Evaluates the derivative of a polynomial with coefficients coeff at input values x.
        ### Input:
        ###   x (Tensor) - Input values.
        ###   coeff (Tensor) - Polynomial coefficients.
        ### Output: Derivative of the polynomial evaluated at input x.
        x = x.reshape([x.shape[0],1])
        x_pre = x ** (torch.arange(coeff.shape[0])-1)
        mult = coeff * torch.arange(coeff.shape[0])
        return x_pre @ mult
    
    def partition_unity(self, x_in):
        ### Partitions the domain into three intervals and returns partition of unit functions associated with each interval.
        ### Input: x_in (Tensor) - Input values.
        ### Output: Tuple containing partition of unity functions evaluated at input x_in.
        x_min = self.epdf.data.min()
        x_max = self.epdf.data.max()
        f1 = lambda x: 1-self.epdf.sigma(self.pot_weight*(x-x_min))
        f2 = lambda x: self.epdf.sigma(self.pot_weight*(x-x_min)) - self.epdf.sigma(self.pot_weight*(x-x_max))
        f3 = lambda x: self.epdf.sigma(self.pot_weight*(x-x_max))
        return f1(x_in), f2(x_in), f3(x_in)
    
    def net_pdf(self,x_in):
        ### Computes the probability density function (PDF) values using the trained density network.
        ### Input: x_in (Tensor) - Input values.
        ### Output: PDF values.
        coeffs = self.params / self.sf 
        p = self.poly_eval(x_in,coeffs)
        p_prime = self.poly_derivative(x_in,coeffs)
        sig = self.activation(p)
        f = sig * (1-sig) * p_prime
        f1, f2, f3 = self.partition_unity(x_in)
        return 0 * f1 + f * f2 + 0 * f3

    def net_cdf(self,x_in):
        ### Computes the cumulative distribution function (CDF) values using the trained density network.
        ### Input: x_in (Tensor) - Input values.
        ### Output: CDF values.
        coeffs = self.params / self.sf 
        p = self.poly_eval(x_in,coeffs)
        sig = self.activation(p)
        f1, f2, f3 = self.partition_unity(x_in)
        return 0 * f1 + sig * f2 + 1 * f3

    def densities_l1_distance(self, x_in, h_values, interval):
        ### Computes the L1 error between predicted and ground truth values and the probability inside the given interval.
        ###   x_in (Tensor) - Input values.
        ###   h_values (Tensor) - Predicted values.
        ###   interval (Tuple) - Interval for which to compute the probability.
        ### Output: L1 error and probability inside the range.
        model_values = self.forward(torch.tensor(x_in, dtype = torch.float32).detach())
        l1_dif_sum = ((torch.abs(h_values - model_values) * (x_in.diff().mean())).sum()).item()
        return l1_dif_sum, self.epdf.prob_interval(x_in, interval)