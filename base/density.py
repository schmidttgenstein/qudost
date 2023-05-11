import torch
import numpy as np 
from scipy.stats import expon
from arch import MLPipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
    

class EPDF():
    def __init__(self,data):
        self.data = data
        self.bw = (data.shape[0]) ** (1/3)


class ECDF():
    def __init__(self, data):
        data.sort()
        self.data = data
        self.N = len(data)

    def domain(self,):
        i=0
        domain=[]
        while i < self.N-1:
            x = (self.data[i]+self.data[i+1])/2
            domain.append(x)
            i+=1
        domain.append(self.data[self.N-1]+1)
        return np.array(domain)

    def ecdf(self, x): 
        idx = self.data <= x
        return 1/(self.N)*idx.sum()
    
    def range_jumps(self,): 
        x = np.linspace(-4,4,self.N)
        F = np.zeros(self.N)
        for j in range(self.N):
            x_j = x[j]
            F[j] = self.ecdf(x_j)
        return x,F
    
class Regression(ECDF):
    def __init__(self, ecdf_values, domain, degree):
        self.ecdf_values = ecdf_values
        self.domain = domain # Save the domain of ECDF
        self.degree = degree

    def support(self,x,F): ## Deleting 0 and 1 to avoid inf results
        #### You should be deleting based on \epsilon and 1 - \epsilon
        compact_idx = (F>0) & (F<1)
        return x[compact_idx],compact_idx
    
    def linear_regression(self,compact, inv):
        pf = PolynomialFeatures(degree=self.degree,include_bias=False)
        lr = LinearRegression()
        ypf = pf.fit_transform(np.array(compact)[:, np.newaxis])
        lr.fit(ypf,inv)
        polynomial = np.insert(lr.coef_, 0, lr.intercept_)
        return lr, polynomial, ypf

def inv_sigmoid(y):
    return np.log(y/(1-y))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def lagrange_pol(t,data):
    '''Function that return the lagrange polynomial that passes exactly through the data
    input: domain (array-linspace) and data (array of x and y)'''
    xdat = data[0]
    ydat = data[1]
    yoft = 0
    for i in range(xdat.size):
        p = ydat[i]
        for j in range(xdat.size):
            if i != j: p *= ((t - xdat[j]) / (xdat[i] - xdat[j]))
        yoft += p 
    return yoft

def poly_eval(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.
    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    #print(f'# This is a polynomial of order {o}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

#Derivative of a polynomial
def derivative(polynomial_coef):
    """ Returns the coefficients of the derivative polynomial
    """
    return np.array([polynomial_coef[i] * i for i in range(1, len(polynomial_coef))])

def simple_pdf(x, polynomial_coef):
    #polyder = np.polynomial.polynomial.polyder(polynomial_coef, m=1)
    polyder = derivative(polynomial_coef)
    pol = poly_eval(x, polynomial_coef)
    return sigmoid(pol)*(1-sigmoid(pol))*poly_eval(x, polyder)


if __name__ == "__main__":
    # create some randomly distributed data:
    N = 10000 # number of datapoints
    k = 7 #degree of polynomial
    
    data = np.random.normal(0,1,N)#expon.rvs(size=N,scale=1)
    #data = np.random.randn(N)

    ecdf = ECDF(data)
    x,F = ecdf.range_jumps()
    reg = Regression(F, data, degree=k)

    #changed def of support() 
    x_compact, compact = reg.support(x,F)
    #and argument in inv_sigmoid
    inv = inv_sigmoid(F[compact])

    model, polynomial_coef, ypf = reg.linear_regression(x_compact, inv)
    yHat = model.predict(ypf)
    xx = np.linspace(0, 1, N)
    
    ## polynomial regression prediction with model
    #pol_pred = model.predict(PolynomialFeatures(degree=k,include_bias=False).fit_transform(xx[:, np.newaxis]))
    
    ## polynomial regression evaluating polynomial (same result)
    pol_pred = poly_eval(x_compact, polynomial_coef)
    
    print('training MSE is: %.5f' % np.mean((inv-yHat)**2))
    print('R-squared is: %.5f' % model.score(ypf,inv)) # R-squared = 1- sum((y-yHat)**2)/sum((y-mean(y))**2)

    ## predicted pdf and cdf
    pred_cdf = sigmoid(pol_pred)
    pred_pdf = simple_pdf(x_compact,polynomial_coef)

   
if __name__ == "__main__":
    a = MLPipeline()
