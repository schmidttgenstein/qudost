import numpy as np
import matplotlib.pyplot as plt
import torch
from qudost.density import ECDF, EPDF, RegressionCDF, DensityNetwork
from qudost.data import DataSet, DataLoader, ImageColorProj
import json
from random import shuffle
from scipy import optimize, stats
import timeit

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


def wasserstein_opt(cdf1, cdf2, x_vals):
    # Compute the cost matrix for the optimal transport problem
    cost_matrix = np.abs(np.subtract.outer(cdf1, cdf2))
    # Solve the assignment problem to get the Wasserstein distance
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
    wasserstein_dist = np.sum(cost_matrix[row_ind, col_ind] * np.diff(x_vals))
    return wasserstein_dist

def wasserstein_cdf(cdf1, cdf2, x_vals):
    wasserstein_distance = torch.trapz(np.abs(cdf1 - cdf2), x_vals)
    return wasserstein_distance

def wasserstein_point(cdf1, point, x_vals):
    heaviside_point = torch.where(x_vals < point, 0, 1)
    wasserstein_distance = torch.trapz(np.abs(cdf1 - heaviside_point), x_vals)
    return wasserstein_distance

def make_regression(x_tr,x_te,deg):
    #ECDF - Linear Regression
    epdf_eval = EPDF(x_te)
    epdf_train = EPDF(x_tr)
    reg = RegressionCDF(epdf_train.cdf, epdf_train.x_domain,degree = deg)
    x,F = epdf_train.filter_cdf(0.00001)
    _,y = epdf_train.sigma_inverse(F)
    model, poly_coeff, ypdf = reg.linear_regression(x,y)
    epdf_eval.coeff = poly_coeff
    epdf_train.coeff = poly_coeff
    return epdf_eval, epdf_train

def fine_tune(epdf_eval, epdf_train, epoch= 100, lr = 0.01, lamb = 0.5):
    dn = DensityNetwork(epdf_train,epoch = epoch,lr = lr, lamb=lamb)
    ds = DataSet(epdf_train.t,epdf_train.h,tor = True,zdim = True)
    dl_tr = DataLoader(ds,batch_size = 1)
    dse = DataSet(epdf_eval.t,epdf_eval.h, tor = True,zdim = True)
    dl_eval = DataLoader(dse,batch_size = 1)
    dn.fit(dl_tr,dl_eval)
    return dn


if __name__ == "__main__":
    patch = str(0) # Ranging from 0 to 149
    class_label = str(0) # Ranging from 0 to 9
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
    deg = 7
    epoch, lr, lamb = 100, 0.01, 0.5
    epdf_eval, epdf_train = make_regression(x_tr, x_te, deg)
    dn = fine_tune(epdf_eval, epdf_train, epoch, lr, lamb)
    dn2 = fine_tune(epdf_train, epdf_eval, epoch, lr, lamb)

    plt.figure(3)
    plt.plot(epdf_train.t,epdf_train.h,'.', label = 'train histo')
    plt.plot(epdf_eval.t,epdf_eval.h,'.',label = 'eval histo')
    x = torch.linspace(epdf_train.t[0],epdf_train.t[-1],490)
    f = dn.forward(torch.tensor(x,dtype = torch.float32).detach())
    f2 = dn2.forward(torch.tensor(x,dtype = torch.float32).detach())
    plt.plot(x,f.detach().numpy(), label = 'training data model')
    #pp = epdf_eval.poly_eval(x,poly_coeff)
    #plt.plot(x,epdf_train.sigma(pp)*(1-epdf_train.sigma(pp))*epdf_train.poly_derivative(x,poly_coeff), label = "LR pdf")
    plt.legend()
    plt.title("Densities with lambda={}, lr={}, for patch {} and label {}".format(lamb, lr, patch, class_label))
    
    plt.figure(4)
    plt.plot(x, f.detach().numpy()/f2.detach().numpy())
    plt.title("Ratio of densities")

    pol1 = epdf_train.poly_eval(x,dn.params.detach().numpy())
    pol2 = epdf_eval.poly_eval(x,dn2.params.detach().numpy())
    plt.figure(5)
    plt.plot(x,f.detach().numpy(), label = 'training data model')
    plt.plot(x,f2.detach().numpy(), label = 'evaluation data model')
    plt.plot(x,dn.activation(pol1), label = "train cdf")
    plt.plot(x,dn2.activation(pol2), label = "evaluation cdf")
    #was = wasserstein_opt(dn.activation(pol1)[:-1],dn.activation(pol2),x)
    cdf1 = dn.activation(pol1)
    cdf2 = dn2.activation(pol2)
    was = wasserstein_cdf(dn.activation(pol1),dn2.activation(pol2),x)
    scipy_time = timeit.timeit(lambda: stats.wasserstein_distance(x_te,x_tr), number=1000)
    custom_time = timeit.timeit(lambda: wasserstein_cdf(cdf1,cdf2,x), number=1000)
    plt.legend()
    plt.title(f"Training and eval w/ Wasserstein distance = {was}")

    #Wasserstein distance point to distribution
    shuffle(data)
    patch_class = torch.tensor(data)
    epdf = EPDF(patch_class)
    x,F = epdf_train.filter_cdf(0.00001)
    wass_dis = wasserstein_point(torch.from_numpy(F), 0.5, x)
    print("Wasserstein distro to point:", wass_dis.item())
    plt.show()

    








