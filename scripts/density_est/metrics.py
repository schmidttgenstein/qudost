import torch
import numpy as np 
import matplotlib.pyplot as plt
from scipy import optimize, stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from qudost.density import ECDF, EPDF, RegressionCDF, DensityNetwork
from qudost.data import DataSet, DataLoader, DataGenerator
from qudost.random import density_projection
from torch.utils.data import DataLoader as TorchDataLoader
from run_density_fit import grad_desc_pdf_tune, regression, gau_mix

def kl_divergence(pdf1, pdf2, x):
    ### Computes the Kullback-Leibler divergence between two distributions.
    ### Input:
    ###   pdf1 (array-like) - PDF values of the first distribution.
    ###   pdf2 (array-like) - PDF values of the second distribution.
    ###   x (array-like) - Input values.
    ### Output:
    ###   kl_div (float) - Kullback-Leibler divergence between the two distributions.
    pdf1 = pdf1 / np.trapz(pdf1,x)
    pdf2 = pdf2 / np.trapz(pdf2,x)
    pdf1 = np.where(pdf1 <= 0, 1e-10, pdf1)
    pdf2 = np.where(pdf2 <= 0, 1e-10, pdf2)
    kl_div = np.trapz(pdf1 * np.log(pdf1 / pdf2),x)
    return kl_div

def js_divergence(pdf1,pdf2,x):
    ### Computes the Jensen-Shannon divergence between two distributions.
    ### Input:
    ###   pdf1 (array-like) - PDF values of the first distribution.
    ###   pdf2 (array-like) - PDF values of the second distribution.
    ###   x (array-like) - Input values.
    ### Output:
    ###   js_div (float) - Jensen-Shannon divergence between the two distributions.
    pdf1 = pdf1 / np.trapz(pdf1,x)
    pdf2 = pdf2 / np.trapz(pdf2,x)
    pdf1 = np.where(pdf1 <= 0, 1e-10, pdf1)
    pdf2 = np.where(pdf2 <= 0, 1e-10, pdf2)
    m = (pdf1 + pdf2) / 2
    js_div = (kl_divergence(pdf1,m,x) + kl_divergence(pdf2,m,x)) / 2
    return js_div

if __name__ == "__main__":
    # (Step 1) Generate data 1 and train a density network
    np.random.seed(20) 
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
    poly_coeff = dn.params.detach().numpy()/scale_factor

    x = torch.linspace(torch.min(x1),torch.max(x1),N)
    #Model PDF
    cdf1 = dn.net_cdf(x).detach().numpy()
    pdf1 = dn.forward(torch.tensor(x,dtype = torch.float32).detach())
    #True PDF
    m1, s1 = gen_dat.gmm_mixtures[0]
    m2, s2 = gen_dat.gmm_mixtures[1]
    pdf2 = stats.norm.pdf(x,m1,s1) + stats.norm.pdf(x,m2,s2)
    pdf2 = pdf2 / np.trapz(pdf2,x)
    
    #GMM PDF
    gmm = gau_mix(epdf_train.data, mix)
    logprob = gmm.score_samples(x.reshape(x.shape[0],1))
    gmm_pdf = np.exp(logprob)
    
    # METRICS
    kl_div = kl_divergence(pdf1.detach().numpy(),pdf2,x)
    print("Model KL Divergence: ", kl_div)

    js_div = js_divergence(pdf1.detach().numpy(),pdf2,x)
    print("Model JS Divergence: ", js_div)

    gmm_kl_div = kl_divergence(gmm_pdf,pdf2,x)
    print("GMM KL Divergence: ", gmm_kl_div)

    gmm_js_div = js_divergence(gmm_pdf,pdf2,x)
    print("GMM JS Divergence: ", gmm_js_div)



