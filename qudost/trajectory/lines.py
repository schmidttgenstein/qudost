import torch 
import copy 
import numpy as np 
import torch.nn as nn
import scipy.linalg as scl 
import torch.optim as optim
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset 

def polynome(t,coeffo):
    coeff = copy.deepcopy(coeffo)
    coeff = np.array(coeff) if type(coeff) is list else coeff
    dim = coeff.shape[0]
    if len(coeff.shape) > 1:
        coeff = coeff.flatten()
    if type(t) is np.ndarray:
        num = t.shape[0]
        t[np.argwhere(np.abs(t)<1e-10)] = 1e-8
        t = t.reshape([num,1])
        ax = 1
    elif torch.is_tensor(t):
        if ~torch.is_tensor(coeff):
            coeff = torch.tensor(coeff)
        num = t.shape[0]
        t[torch.argwhere(torch.abs(t)<1e-10)] = 1e-8
        t = t.reshape([num,1])
        ax = 1
    else: 
        num = 1
        ax  = None
        t = 1e-10 if t == 0 else t 
    result = np.zeros([num,dim])
    for j in np.arange(dim):
        exp = np.arange(dim) - j
        result[:,j] = (coeff * (t ** exp)).sum(axis = ax)
        coeff *= exp 
    return result

def derivative(p,fun,theta = None,n=1,h=1e-2,axis = None,r_all = False):
    pm = p-h/2 
    pp = p+h/2
    if n > 0:
        out = (derivative(pp,fun,theta,n-1,h,axis,r_all=False) - derivative(pm,fun,theta,n-1,h,axis,r_all=False))/h
    else:
        out = fun(p,theta,r_all)
    return out

def derivatives(p,fun,theta,h=1e-2,n=1):
    if torch.is_tensor(p):
        data = torch.zeros([p.shape[0],n])
    else: 
        data =np.zeros([p.shape[0],n])
    for j in np.arange(n):
        if j == 0:
            result,other = derivative(p,fun,theta,j,h,r_all=True)
        else:
            result = derivative(p,fun,theta,j,h)
        data[:,j] = result.flatten()
    return data, other

class BaseIntegrator:
    def __init__(self,function,dim = None):
        self.integrator = function
        self.dim        = dim 
        
    def integrate(self,t,x0,num_mesh=None):
        if num_mesh is not None:
            t = torch.linspace(0,t,num_mesh) 
        return self.integrator(t,x0)
    

class TrajData(Dataset):
    def __init__(self,x,y,tx0):
        self.x = x
        self.y = y 
        self.ohe_dim = self.x.shape[1]
        self.dim = tx0[0,1:].shape[0]
        self.len = y.shape[0]
        self.tx = tx0 

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        return self.x[idx,:],self.tx[idx,0], self.tx[idx,1:], self.y[idx,:]
    
    def get_metadata(self,):
        return {'dim':self.dim,'embed_dim':self.ohe_dim,'len':self.len}

def generate_data(ild,m_traj=50,n_time=50000,m_tot = int(5e4),mu_dat=None,sig_dat=None,check=False):
    #upto_deg: how many derivatives do you want?
    return_ext = False
    dim = ild.dim #if upto_deg is not None else upto_deg
    mu = np.zeros(dim)
    sig = np.eye(dim)*(10**(dim-np.arange(dim)-1))
    x0_in = np.random.multivariate_normal(mu,sig,m_traj)*2.5
    t1_in = np.random.normal(0,1,n_time)*50
    t_idx, x_idx = np.random.randint(0,n_time,m_tot), np.random.randint(0,m_traj,m_tot)
    t_dat, x_dat = t1_in[t_idx], x0_in[x_idx]
    tx_data = np.concatenate((t_dat.reshape(t_dat.shape[0],1),x_dat),axis = 1)
    y_data = np.zeros([m_tot,dim])
    x_ohe = np.zeros([m_tot,m_traj])
    for idx in range(m_tot):  #for d_idx in range(upto_deg+1):
        y_data[idx,:] =ild.integrate(t_dat[idx],x_dat[idx,:]) 
        x_ohe[idx,:] = ohe(x_idx[idx],m_traj)
    if mu_dat is None:
        mu_dat = y_data.mean(axis = 0)
        sig_dat = y_data.std(axis = 0)
        return_ext = True
    #y_data = (y_data - mu_dat)/sig_dat  
    if check: 
        for j in range(m_traj):
            idx = x_idx == j
            t_test = t_dat[idx]
            y_test = y_data[idx]
            if j < 3:
                plt.plot(t_test,y_test,'.',label=j)
        plt.legend()
        plt.show()
    if return_ext:
        return torch.tensor(x_ohe,dtype = torch.float32) ,torch.tensor(y_data,dtype=torch.float32),torch.tensor(tx_data,dtype=torch.float32), mu_dat, sig_dat
    else:
        return torch.tensor(x_ohe,dtype = torch.float32) ,torch.tensor(y_data,dtype=torch.float32),torch.tensor(tx_data,dtype=torch.float32) #,torch.tensor(z_data,dtype = torch.float32)


class TrajCD(nn.Module):
    def __init__(self,loader,integrator,epochs = 15, lr = .01,mu_dat = None,sig_dat = None):
        super().__init__()
        self.epochs = epochs 
        self.lr = lr
        meta_dat = loader.dataset.get_metadata()
        self.bs = loader.batch_size
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.integrator = integrator
        ohe_dim = meta_dat["embed_dim"] 
        dim = meta_dat["dim"]
        #TODO handle dimensionality + constraints
        self.encode = nn.Sequential(nn.Linear(ohe_dim,dim))
        self.propagate = nn.Sequential(nn.Linear(1+dim,150),nn.Sigmoid(), nn.Linear(150,1)) 
        self.opt = optim.Adam(self.parameters(),lr = self.lr)
        self.mu_dat = torch.tensor(mu_dat) 
        self.sig_dat = torch.tensor(sig_dat)
        self.sig_dat[self.sig_dat<1] = 1.
        self.dalpha = self.bs/(meta_dat["len"]*epochs) - 1e-6
        self.dim = dim 

    def metrics(self,y_score,y_truth):
        with torch.no_grad():
            _, y_pred = y_score.max(axis = 1)
            acc = (y_pred == y_truth).float().mean().item()
        return  acc
    
    def loss(self,enc_x,x_in,ph_tx,y_truth,alpha = 0.75):
        el1 = (enc_x - x_in)
        el2 = torch.abs(ph_tx - y_truth)# / self.sig_dat
        alpha = 1
        return (1-alpha)* torch.norm(el1,dim=1).mean() +  torch.norm(el2,dim = 1).mean() #alpha*el2[:,0].mean()# @(1.0**torch.linspace(0,self.dim,self.dim))
    
    def train_step(self,x_emb,t_in,x_0,y_truth,alpha=.75):
        ph_tx, decenc = derivatives(t_in,self.forward,x_0,n=y_truth.shape[1]) #delete
        #ph_tx, decenc = derivatives(t_in,self.forward,x_emb,n=y_truth.shape[1])
        el = self.backward(decenc,x_0,ph_tx,y_truth,alpha)
        self.update()
        return el 

    def forward(self,t_in,x_in,r_all = False):
        #enc_x = self.encode(x_in)
        enc_x = x_in #delete
        if t_in.numel() > 1:
            tx = torch.cat((t_in.reshape([t_in.shape[0],1]),enc_x),dim = 1)
        else:
            tx = torch.cat((t_in.reshape(1),enc_x))
        phi_tx = self.propagate(tx)
        if r_all: 
            return phi_tx, enc_x,  
        else: 
            return phi_tx
    
    def backward(self,dx,x_0,ph_tx,y_truth,alpha):
        self.opt.zero_grad()
        loss = self.loss(dx,x_0,ph_tx,y_truth,alpha=alpha)
        loss.backward()
        with torch.no_grad():
            return loss.item()

    def update(self,grad = None):
        self.opt.step()

    def fit(self,train_loader,validation_loader=None,plot=False):
        alpha = 0 
        for epoch in range(self.epochs):
            print(f"on epoch {epoch}")
            print("---------------")
            for batch_idx,data in enumerate(train_loader):
                x_embed, time, init_cond, output = data[0], data[1], data[2], data[3]
                it_loss = self.train_step(x_embed,time,init_cond,output,alpha)
                print(f"batch {batch_idx} with loss {it_loss:.4f}")
                alpha += self.dalpha
            if validation_loader is not None:
                pass 
            if plot:
                self.eval 
        print(f"with final alpha {alpha}")

    def precheck(self,dataloader):
        ds = dataloader.dataset
        idx = torch.zeros(ds.__len__(),dtype=torch.bool)
        t=[]
        y = []
        for j in range(ds.__len__()):
            coeff = ds[0][2]
            if (ds[j][0] == ds[0][0]).all():
                idx[j] = True
                t.append(ds[j][1].item())
                y.append(ds[j][3].tolist())
        y = np.array(y)
        t = np.array(t)
        plt.figure(4)
        plt.plot(t,y[:,0],'.')
        tsp = np.linspace(t.min(),t.max(),100)
        yout =  self.integrator.integrate(tsp,coeff)
        plt.plot(tsp,yout[:,0])
        #plt.show()
 

    def eval(self,x0,coeff,t_end,n_step = 100):
        t_in = torch.linspace(0,t_end,n_step)
        x0 = x0.repeat(n_step).reshape([n_step,x0.shape[0]])
        coeff =coeff.repeat(t_in.shape).reshape([t_in.shape[0],coeff.shape[0]]) #delete
        #tx = torch.cat((t_in.reshape([t_in.shape[0],1]),x0.repeat(n_step).reshape([n_step,self.integrator.dim])),dim = 1)
        with torch.no_grad():
            #x1_model, x_enc = derivatives(t_in,self.forward,x0,n=2)
            x1_model, _ = derivatives(t_in,self.forward,coeff,n=2)
            x_enc = coeff  #delete
            coeff = coeff[0,:]
            x1_prop = torch.tensor(self.integrator.integrate(t_end,x_enc[0,:].detach().numpy(),n_step),dtype=torch.float32)
            x1_truth = torch.tensor(self.integrator.integrate(t_end,coeff,n_step),dtype=torch.float32)
        return t_in.detach().numpy(), x1_truth.detach().numpy(), x1_prop.detach().numpy(), x1_model.detach().numpy()
    
    def postcheck(self,loader,idx_num = 1):
        ev_data = loader.dataset
        for idx in range(idx_num):
            xe0 = ev_data[idx][0]
            coeff = ev_data[idx][2]
            a,b,c,d = self.eval(xe0,coeff,50)
            batch = next(enumerate(loader))
            #_, x_enc = self.forward(batch[1][1],batch[1][0],r_all = True)
            x_enc = batch[1][2] #delete
            plt.figure(1)
            plt.plot(a,b[:,0],label = f'traj {idx} truth')
            #plt.plot(a,c[:,0],'-.',label = 'prop enc')
            plt.plot(a,d[:,0],'--',label = f'traj {idx} model')
            plt.legend()
            plt.title('eval')
            plt.figure(2)
            plt.plot(a,b[:,1],label = f'traj {idx} truth')
            #plt.plot(a,c[:,1],'-.',label = 'prop enc')
            plt.plot(a,d[:,1],'--',label = f'traj {idx} model')
            plt.legend()
            plt.title('derivative')
        plt.figure(3)
        plt.plot(batch[1][1].detach().numpy(),torch.norm(batch[1][2]-x_enc,dim=1).detach().numpy(),'.')
        plt.title('encoder error')
        #self.precheck(loader)
        plt.show()
        


def ohe(idx,dim):
    return np.eye(1,dim,idx)



if __name__ == "__main__":
    stable = False
    bs = 5000
    N_data = int(2.5e5)
    N_init = 25000
    n_time = 5000
    N_data_eval = int(1e3)

    propo = BaseIntegrator(polynome,dim = 3)


    in_data, out_data,tx0,mu_dat,sig_dat = generate_data(propo,m_traj = N_init,n_time = n_time, m_tot = N_data)
    in_data_eval, out_data_eval,tx0_eval, = generate_data(propo,m_traj = N_init,n_time = n_time, m_tot = N_data_eval,mu_dat=mu_dat,sig_dat=sig_dat,check= False )

    TrainData = TrajData(in_data,out_data,tx0)
    EvalData = TrajData(in_data_eval,out_data_eval,tx0)
    TrajLoader = DataLoader(TrainData,batch_size = bs,shuffle = True)
    EvalLoader = DataLoader(EvalData,batch_size = bs,shuffle = False)
    

    eta = 2.5e-1
    tcd = TrajCD(TrajLoader,propo,epochs = 150,lr = eta,mu_dat=mu_dat,sig_dat=sig_dat)

    #tcd.precheck(TrajLoader)
    tcd.fit(TrajLoader,plot=True)
    tcd.postcheck(TrajLoader,idx_num = 5)
    #tcd.postcheck(EvalLoader,idx_num = 30)
    #t,x1,x2 = tcd.eval(x0,10)
    # plot_sample(t,x1,x2,'Eval Data')
    #t,x1,x2 = tcd.eval(data0[0],10)
    # plot_sample(t,x1,x2,title = 'Train Data')
