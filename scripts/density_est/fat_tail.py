import numpy as np 
import matplotlib.pyplot as plt


def inv_cdf(y):
    u = np.zeros(y.shape[0])
    idx1 =  y<=1/4
    u[idx1] =  -1/(4*y[idx1])
    idx2 = (y>1/4) & (y<=3/4)
    u[idx2] = 4*y[idx2]-2
    idx3 = y>3/4
    u[idx3] = 1/(4*(1-y[idx3]))
    return u 

def gen_data(n):
    data = np.random.random(n)
    return inv_cdf(data)


if __name__ == "__main__":
    N = 10000000
    d = gen_data(N)
    plt.hist(d[(d>-200)&(d<200)],bins = 1000,density = True)
    plt.show()
