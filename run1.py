from efficient_kan import KAN
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import numpy as np
from jax import random, jit
import sklearn
from sklearn.preprocessing import MinMaxScaler
import itertools
from functools import partial
from tqdm import trange, tqdm
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import time
from torch.func import jacrev, hessian
start_time=time.time()
device = torch.device("cuda")
class PI_DeepONet(nn.Module):
    def __init__(self,model1,model2,model3,model4,model5):
        super(PI_DeepONet, self).__init__()
        # Network initialization and evaluation functions
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5


        self. bc_losses = []
        self. pde_losses = []
    def reshape(self,X):
        reshaped_X = X.reshape(-1,)
        return reshaped_X
    def brunk_net(self,u1,u2,u3):
        BC1=self.model1(u1)
        BC2=self.model2(u2)
        BC3 = self.model3(u3)
        B=BC1*BC2*BC3
        return B


    def helper(self,X, Y):
        reshaped_X=self.reshape(X)
        reshaped_Y=self.reshape(Y)
        stacked_tensor = torch.stack([reshaped_X, reshaped_Y])
        permuted_tensor = stacked_tensor.permute(1, 0)
        return permuted_tensor

    # Define DeepONet architecture
    def operator_net(self,u1,u2,u3,x,t):
        n=len(x)
        B1=self.brunk_net(u1,u2,u3)
        B1= (B1.view(1, -1)).repeat(n, 1).float()
        B = self.model4(B1)
        y = self.helper(x, t)
        T = self.model5(y)
        outputs =torch.sum(B * T, dim=1)
        return outputs

    # Define PDE residual
    def residual_net(self,u1,u2,u3,x,t):
        s=self.operator_net(u1,u2,u3,x,t)
        s_x =jacrev(self.operator_net,argnums=3)(u1,u2,u3,x,t).sum(dim=0)
        s_xx =(hessian(self.operator_net,argnums=3)(u1,u2,u3,x,t).sum(dim=0)).sum(dim=0)
        s_t =jacrev(self.operator_net,argnums=4)(u1,u2,u3,x,t).sum(dim=0)
        res =s_t-(1/2)*(0.165856529**2)*(x**2)*s_xx-0.0256*x*s_x+0.0256*s
        return res
        # r =0.025610
        # v=0.165856529

    # Define boundary loss
    def loss_bcs(self,u1,u2, u3,x, t, output):
        # Compute forward pass
        s_pred = self.operator_net(u1,u2,u3,x,t)
        # Compute loss
        loss = torch.mean((output.flatten() - s_pred) ** 2)
        return loss


    # Define residual loss
    def loss_res(self,u1,u2,u3,x, t, output):
        # Compute forward pass
        pred = self.residual_net(u1,u2,u3,x,t)
        loss = torch.mean((output.flatten() - pred) ** 2)
        return loss



    def train(self,u1,u2,u3,dataloader1,dataloader2):
        params1 = tuple(model1.parameters())
        params2 = tuple(model2.parameters())
        params3 = tuple(model3.parameters())
        params4 = tuple(model4.parameters())
        params5 = tuple(model5.parameters())
        params = params1 + params2+ params3+ params4+ params5
        # params = (model1.parameters(), model2.parameters())
        # Initialize optimizer

        self.optimizer = torch.optim.LBFGS(params, lr=1)
        pbar = tqdm(range(200), desc='description')
       
        for _ in pbar:
           
            
            for (x_i, t_i,outputs_i),(x_b, t_b, outputs_b) in zip(dataloader1, dataloader2):
                def closure():
                    global pde_loss, bc_loss
                    self.optimizer.zero_grad()
                    bc_loss= self.loss_bcs(u1,u2,u3, x_i, t_i,outputs_i)
                    pde_loss=self.loss_res(u1,u2,u3,x_b, t_b, outputs_b)
                    loss =pde_loss+50*bc_loss
                    loss.backward()
                    return loss

            # if _ % 5 == 0 and _ < 50:
                # model1.update_grid_from_samples(u_i1)
                # model1.update_grid_from_samples(u_i2)
                # model1.update_grid_from_samples(u_b1)
                # model1.update_grid_from_samples(u_b2)
                self.optimizer.step(closure)
       
           

            if _ % 1 == 0:
                pbar.set_description("pde loss: %.2e | bc loss1: %.2e" % (
                pde_loss.detach().cpu().numpy(), bc_loss.detach().cpu().numpy()))

            self.pde_losses.append(pde_loss.detach().cpu().numpy())
            self.bc_losses.append(bc_loss.detach().cpu().numpy())








# Deinfe initial and boundary conditions for advection equation
def f1(x,t,k):
  return np.where(t==0,np.maximum(x-k,0),0)
def f2(x,k):
  return np.where(x ==3*k, x-k, 10)
def f3(x):
  return np.where(x==0,0,0)




def min_max_normalize(x, min_val, max_val):
    normalized_x = (x - min_val) / (max_val - min_val)
    return normalized_x


# Geneate training data corresponding to one input sample
def generate_one_training_data(key,P,Q,K):
    subkeys = random.split(key, 10)

    np_K=K*(np.ones((P // 3, 1)))

    x_bc1 = random.uniform(subkeys[2], shape=(P // 3, 1), minval=0, maxval=3* K)
    x_bc2 = 3 * K * (np.ones((P // 3, 1)))
    x_bc3 = np.zeros((P // 3, 1))
    x_bcs = np.vstack([x_bc1, x_bc2,x_bc3])
    x_bcs_min_value = np.min(x_bcs)
    x_bcs_max_value = np.max(x_bcs)
    x_bcs=min_max_normalize(x_bcs, x_bcs_min_value, x_bcs_max_value)
    x_bcs= x_bcs.__array__()
    x_i = torch.tensor(x_bcs)

    t_bc1 = np.zeros((P // 3, 1))
    t_bc2 = random.uniform(subkeys[3], shape=(P // 3, 1), minval=0, maxval=365)
    t_bc3 = random.uniform(subkeys[4], shape=(P // 3, 1), minval=0, maxval=365)
    t_bcs = np.vstack([t_bc1, t_bc2,t_bc3])
    t_bcs_min_value = np.min(t_bcs)
    t_bcs_max_value = np.max(t_bcs)
    t_bcs= min_max_normalize(t_bcs, t_bcs_min_value,t_bcs_max_value)
    t_bcs = t_bcs.__array__()
    t_i = torch.tensor(t_bcs)


    s_bc1=f1(x_bc1,t_bc1,np_K)
    s_bc1 =np.array(list(s_bc1))
    s_bc1 =s_bc1.reshape(-1,1)
    s_bc2 = f2(x_bc2,np_K)
    s_bc2 = np.array(list(s_bc2))
    s_bc2 = s_bc2.reshape(-1, 1)
    s_bc3 = f3(x_bc3)
    s_bc3 = np.array(list(s_bc3))
    s_bc3 = s_bc3.reshape(-1, 1)
    s_train= np.vstack([s_bc1, s_bc2,s_bc3])
    s_bcs_min_value = np.min(s_train)
    s_bcs_max_value = np.max(s_train)
    s_train= min_max_normalize(s_train,s_bcs_min_value, s_bcs_max_value)
    s_train= s_train.__array__()

    outputs_i= torch.tensor(s_train)

    x_b = random.uniform(subkeys[4], shape=(Q, 1), minval=0, maxval=3*K)
    t_b = random.uniform(subkeys[5], shape=(Q, 1), minval=0, maxval=365)
    x_b = min_max_normalize(x_b,x_bcs_min_value, x_bcs_max_value)
    t_b= min_max_normalize(t_b,t_bcs_min_value,t_bcs_max_value)
    x_b = x_b.__array__()
    x_b= torch.tensor(x_b)
    t_b = t_b.__array__()
    t_b = torch.tensor(t_b)
    outputs_b = torch.zeros((Q, 1))

    x_bc11=min_max_normalize(x_bc1, x_bcs_min_value, x_bcs_max_value)
    x_bc11=x_bc11.__array__()
    x_bc11= torch.tensor(x_bc11)
    t_bc11 = min_max_normalize(t_bc1, t_bcs_min_value, t_bcs_max_value)
    t_bc11 = t_bc11.__array__()
    t_bc11 = torch.tensor(t_bc11)
    s_bc11 = min_max_normalize(s_bc1, s_bcs_min_value, s_bcs_max_value)
    s_bc11= s_bc11.__array__()
    s_bc11 = torch.tensor(s_bc11)
    u_1= torch.cat((x_bc11,t_bc11,s_bc11), dim=1)

    x_bc22 = min_max_normalize(x_bc2, x_bcs_min_value, x_bcs_max_value)
    x_bc22 = x_bc22.__array__()
    x_bc22 = torch.tensor(x_bc22)
    t_bc22 = min_max_normalize(t_bc2, t_bcs_min_value, t_bcs_max_value)
    t_bc22 = t_bc22.__array__()
    t_bc22 = torch.tensor(t_bc22)
    s_bc22 = min_max_normalize(s_bc2, s_bcs_min_value, s_bcs_max_value)
    s_bc22 = s_bc22.__array__()
    s_bc22 = torch.tensor(s_bc22)
    u_2 = torch.cat((x_bc22, t_bc22, s_bc22), dim=1)


    x_bc33 = min_max_normalize(x_bc3, x_bcs_min_value, x_bcs_max_value)
    x_bc33 = x_bc33.__array__()
    x_bc33 = torch.tensor(x_bc33)
    t_bc33= min_max_normalize(t_bc3, t_bcs_min_value, t_bcs_max_value)
    t_bc33 = t_bc33.__array__()
    t_bc33= torch.tensor(t_bc33)
    s_bc33 = min_max_normalize(s_bc3, s_bcs_min_value, s_bcs_max_value)
    s_bc33 = s_bc33.__array__()
    s_bc33 = torch.tensor(s_bc33)
    u_3 = torch.cat((x_bc33, t_bc33, s_bc33), dim=1)
    # print('x', x_bc33)
    # print(t_bc33)
    # print(s_bc33)

    return u_1,u_2,u_3,x_i,t_i,outputs_i,x_b,t_b,outputs_b, \
           s_bcs_min_value, s_bcs_max_value,x_bcs_min_value, x_bcs_max_value,t_bcs_min_value, t_bcs_max_value



key = random.PRNGKey(0)

K=2.411
P =450 # number of output sensors, 100 for each side
Q =200  # number of collocation points for each input sample
u_1,u_2,u_3,x_i, t_i,outputs_i, x_b, t_b, outputs_b,\
         s_bcs_min_value, s_bcs_max_value,x_bcs_min_value, x_bcs_max_value,t_bcs_min_value, t_bcs_max_value\
            =generate_one_training_data(key,P,Q,K)
u_1=u_1.float().to(device)
# print(u_1)
u_2=u_2.float().to(device)
u_3=u_3.float().to(device)
x_i=x_i.float()
t_i=t_i.float()
outputs_i=outputs_i.float()
x_b=x_b.float()
t_b=t_b.float()
outputs_b=outputs_b.float()
x_i = x_i.reshape(-1,)
t_i = t_i.reshape(-1,)
outputs_i = outputs_i.reshape(-1,)
x_b = x_b.reshape(-1,)
t_b = t_b.reshape(-1,)
outputs_b = outputs_b.reshape(-1,)
dataset1 = TensorDataset(x_i,t_i,outputs_i)
dataset2 = TensorDataset(x_b,t_b,outputs_b)
batch_size1= 45
batch_size2= 20
dataloader1 = DataLoader(dataset1, batch_size=batch_size1, shuffle=True)
dataloader2 = DataLoader(dataset2, batch_size=batch_size2, shuffle=True)



model1 =KAN([3, 5, 1], base_activation=nn.Identity)
model2 = KAN([3, 5, 1], base_activation=nn.Identity)
model3 = KAN([3, 5, 1], base_activation=nn.Identity)
model4 = KAN([150, 5, 5], base_activation=nn.Identity)
model5 = KAN([2, 5, 5], base_activation=nn.Identity)
model= PI_DeepONet(model1,model2,model3,model4,model5)
model.to(device)
model.train(u_1,u_2,u_3,dataloader1,dataloader2)
data=pd.read_csv('data.csv')
x_test=data.iloc[:,1]
t_test=data.iloc[:,2]
x_test=torch.tensor(x_test).float()
x_test=x_test.unsqueeze(1).to(device)
t_test=torch.tensor(t_test).float()
t_test=t_test.unsqueeze(1).to(device)

x_test=min_max_normalize(x_test,x_bcs_min_value, x_bcs_max_value)
t_test=min_max_normalize(t_test,t_bcs_min_value,t_bcs_max_value)

s_pred = model.operator_net(u_1,u_2,u_3,x_test,t_test)
s_pred=s_pred*s_bcs_max_value
s_true=data.iloc[:,3]
s_true=torch.tensor(s_true).to(device)
error_s =(s_pred- s_true)/s_true
print('s_pred:\n',s_pred)
print('s_true:\n',s_true)
print('error_s:\n',error_s)
end_time=time.time()
rap_time=end_time-start_time
print('run-time:{}'.format(rap_time))
