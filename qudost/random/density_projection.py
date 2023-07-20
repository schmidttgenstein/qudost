import numpy as np
import matplotlib.pyplot as plt
import torch
from qudost.density import ECDF, EPDF, RegressionCDF, DensityNetwork
from qudost.data import DataSet, DataLoader, ImageColorProj
import json
from random import shuffle

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    patch = str(123) # Ranging from 0 to 149
    class_label = str(1) # Ranging from 0 to 9
    #print(patch_0_feature_values)
    path = "feature_values_by_patches.json"
    feature_values_by_patches = read_json_file(path)
    data = feature_values_by_patches[patch][class_label]
    shuffle(data)
    patch_class = torch.tensor(data)
    plt.figure(1)
    plt.hist(patch_class,bins = 75,density = True, label = 'class {}'.format(class_label))
    plt.title("Histogram")
    plt.show()

    split = 0.5
    split_idx = int(split * patch_class.shape[0])
    x_tr = patch_class[:split_idx]
    x_te = patch_class[split_idx:]

    #ECDF - Linear Regression
    epdf_eval = EPDF(x_te)
    epdf_train = EPDF(x_tr)
    deg = 7
    reg = RegressionCDF(epdf_train.cdf, epdf_train.x_domain,degree = deg)
    x,F = epdf_train.filter_cdf(0.00001)
    _,y = epdf_train.sigma_inverse(F)
    model, poly_coeff, ypdf = reg.linear_regression(x,y)
    epdf_eval.coeff = poly_coeff
    epdf_train.coeff = poly_coeff
    p = epdf_train.poly_eval(x,poly_coeff)

    plt.figure(2)
    plt.plot(x,y, label = 'sigma inverse cdf')
    plt.plot(x,p, label = 'polynomial')
    plt.title("Sigma inverse regression")
    plt.legend()
    plt.show()

    # Gradient Descent Correction
    epoch, lr, lamb = 200, 0.01, 0.5
    dn = DensityNetwork(epdf_train,epoch = epoch,lr = lr, lamb=lamb)
    ds = DataSet(epdf_train.t,epdf_train.h,tor = True,zdim = True)
    dl_tr = DataLoader(ds,batch_size = 1)
    dse = DataSet(epdf_eval.t,epdf_eval.h, tor = True,zdim = True)
    dl_eval = DataLoader(dse,batch_size = 1)
    dn.fit(dl_tr,dl_eval)

    plt.figure(3)
    plt.plot(epdf_train.t,epdf_train.h,'.', label = 'train histo')
    plt.plot(epdf_eval.t,epdf_eval.h,'.',label = 'eval histo')
    f2 = dn.forward(torch.tensor(epdf_train.t,dtype = torch.float32).detach())
    plt.plot(epdf_train.t,f2.detach(),label = 'model pred hist')
    pp2 = epdf_eval.poly_eval(epdf_train.t,poly_coeff)
    plt.plot(epdf_train.t,epdf_train.sigma(pp2)*(1-epdf_train.sigma(pp2))*epdf_train.poly_derivative(epdf_train.t,poly_coeff), label = "LR pred hist")
    x = torch.linspace(epdf_train.t[0],epdf_train.t[-1],1000)
    f = dn.forward(torch.tensor(x,dtype = torch.float32).detach())
    plt.plot(x,f.detach().numpy(), label = 'actual model')
    pp = epdf_eval.poly_eval(x,poly_coeff)
    plt.plot(x,epdf_train.sigma(pp)*(1-epdf_train.sigma(pp))*epdf_train.poly_derivative(x,poly_coeff), label = "LR pdf")
    plt.legend()
    plt.title("Densities with lambda={}, lr={}, for patch {} and label {}".format(lamb, lr, patch, class_label))
    plt.show()








