from kan import KAN, LBFGS
import torch
import matplotlib.pyplot as plt
from torch import autograd
from tqdm import tqdm
import jax
import torch
import numpy as np
# import jax.numpy as jnp
from jax import random, jit, hessian
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from jax.nn import relu, elu
from jax.config import config
from jax.ops import index_update, index
from jax import lax
from jax.flatten_util import ravel_pytree
import sklearn
from sklearn.preprocessing import MinMaxScaler
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
import pandas as pd
from torch.cuda.amp import GradScaler
import time
from functorch import make_functional, vmap, jacrev, hessian
start_time=time.time()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
class PI_DeepONet:
    def __init__(self,model1,model2,model3):
        # Network initialization and evaluation functions
        self.model1 = model1
        self.model2=model2
        self.model3 = model3
        # self.model4 = model4


        self. bc_losses = []
        self. pde_losses = []

    def to(self, device):
        self.model1.to(device)
        self.model2.to(device)
        self.model3.to(device)
        # self.model4.to(device)
    def reshape(self,X):
        reshaped_X = X.reshape(-1,)
        return reshaped_X

    def helper(self,X, Y):
        reshaped_X=self.reshape(X)
        reshaped_Y=self.reshape(Y)
        stacked_tensor = torch.stack([reshaped_X, reshaped_Y])
        permuted_tensor = stacked_tensor.permute(1, 0)
        return permuted_tensor

    # Define DeepONet architecture
    def operator_net(self,u1,u2,x,t):
        y= self.helper(x,t)
        B1 = self.model1(u1)
        B2 = self.model2(u2)
        B=B1*B2
        T = self.model3(y)
        outputs =torch.sum(B * T, dim=1)
        return outputs

    # Define PDE residual
    def residual_net(self,u1,u2,x,t):
        s=self.operator_net(u1,u2,x,t)
        s_x =jacrev(self.operator_net,argnums=2)(u1,u2,x,t).sum(dim=0)
        s_xx =hessian(self.operator_net,argnums=2)(u1,u2,x ,t)[2].sum(dim=0).sum(dim=0)
        print(s_xx.shape)
        s_t =jacrev(self.operator_net,argnums=3)(u1,u2,x,t).sum(dim=0)
        res =s_t-(1/2)*(0.148**2)*(x**2)*s_xx-0.02266*x*s_x+0.02266*s
        return res
        # 0.02023

    # Define boundary loss
    def loss_bcs(self,u1,u2, x, t, output):
        # Compute forward pass
        s_pred = self.operator_net(u1,u2,x,t)
        # Compute loss
        loss = torch.mean((output.flatten() - s_pred) ** 2)
        return loss


    # Define residual loss
    def loss_res(self,u1,u2,x, t, output):
        # Compute forward pass
        pred = self.residual_net(u1,u2,x,t)
        loss = torch.mean((output.flatten() - pred) ** 2)
        return loss



    def train(self,u_i1,u_i2, x_i, t_i,outputs_i,u_b1,u_b2,x_b, t_b, outputs_b):

        params1 = tuple(model1.parameters())
        params2 = tuple(model2.parameters())
        params3 = tuple(model3.parameters())
        params = params1 + params2+ params3
        # params = (model1.parameters(), model2.parameters())
        # Initialize optimizer

        self.optimizer = LBFGS(params, lr=10, history_size=10, line_search_fn="strong_wolfe",
                               tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
        pbar = tqdm(range(5), desc='description')
        scaler = GradScaler()
        for _ in pbar:

            def closure():
                global pde_loss, bc_loss
                self.optimizer.zero_grad()
                bc_loss= self.loss_bcs(u_i1,u_i2, x_i, t_i,outputs_i)
                pde_loss=self.loss_res(u_b1,u_b2,x_b, t_b, outputs_b)
                loss =pde_loss+100*bc_loss
                loss.backward()
                return loss

            # if _ % 5 == 0 and _ < 50:
                # model1.update_grid_from_samples(u_i1)
                # model1.update_grid_from_samples(u_i2)
                # model1.update_grid_from_samples(u_b1)
                # model1.update_grid_from_samples(u_b2)
            self.optimizer.step(closure)
            scaler.update()


            if _ % 1 == 0:
                pbar.set_description("pde loss: %.2e | bc loss1: %.2e" % (
                pde_loss.detach().numpy(), bc_loss.detach().numpy()))

            self.pde_losses.append(pde_loss.detach().numpy())
            self.bc_losses.append(bc_loss.detach().numpy())








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
    t_bc2 = random.uniform(subkeys[3], shape=(P // 3, 1), minval=0, maxval=248)
    t_bc3 = random.uniform(subkeys[4], shape=(P // 3, 1), minval=0, maxval=248)
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
    s_bc1= min_max_normalize(s_bc1,s_bcs_min_value, s_bcs_max_value)
    s_bc2= min_max_normalize(s_bc2,s_bcs_min_value, s_bcs_max_value)
    s_train= min_max_normalize(s_train,s_bcs_min_value, s_bcs_max_value)
    s_train= s_train.__array__()

    outputs_i= torch.tensor(s_train)

    x_b = random.uniform(subkeys[4], shape=(Q, 1), minval=0, maxval=3*K)
    t_b = random.uniform(subkeys[5], shape=(Q, 1), minval=0, maxval=248)
    x_b = min_max_normalize(x_b,x_bcs_min_value, x_bcs_max_value)
    t_b= min_max_normalize(t_b,t_bcs_min_value,t_bcs_max_value)
    x_b = x_b.__array__()
    x_b= torch.tensor(x_b)
    t_b = t_b.__array__()
    t_b = torch.tensor(t_b)
    outputs_b = torch.zeros((Q, 1))


    u1 = s_bc1.__array__()
    u1 = torch.tensor(u1)
    u2 = s_bc2.__array__()
    u2= torch.tensor(u2)

    return u1,u2,x_i,t_i,outputs_i,x_b,t_b,outputs_b, \
           s_bcs_min_value, s_bcs_max_value,x_bcs_min_value, x_bcs_max_value,t_bcs_min_value, t_bcs_max_value



key = random.PRNGKey(0)

K=2.411
P =300 # number of output sensors, 100 for each side
Q = 100  # number of collocation points for each input sample
u1,u2, x_i, t_i,outputs_i, x_b, t_b, outputs_b,\
 s_bcs_min_value, s_bcs_max_value,x_bcs_min_value, x_bcs_max_value,t_bcs_min_value, t_bcs_max_value\
            =generate_one_training_data(key,P,Q,K)
u2=u2[0]
# print(u2)
u_i1=u1.view(1, -1)
u_i1=u_i1.repeat(P, 1).float().to(device)
u_i2=u2.view(1, -1)
u_i2=u_i2.repeat(P, 1).float().to(device)
x_i=x_i.float()
t_i=t_i.float().to(device)
outputs_i=outputs_i.float().to(device)
u_b1=u1.view(1, -1)
u_b1=u_b1.repeat(Q, 1).float().to(device)
u_b2=u2.view(1, -1)
u_b2=u_b2.repeat(Q, 1).float().to(device)
x_b=x_b.float().to(device)
t_b=t_b.float().to(device)
outputs_b=outputs_b.float().to(device)

print(u_i1)
print(x_i.dtype)
print(x_b)
print(u_i2.dtype)


model1 = KAN(width=[100,5,5], grid=1, k=3, grid_eps=1.0, noise_scale_base=0.25)
model2 = KAN(width=[1,5,5], grid=1, k=3,grid_eps=1.0, noise_scale_base=0.25)
model3 = KAN(width=[2,5,5], grid=1, k=3, grid_eps=1.0, noise_scale_base=0.25 )
    # Create an instance of the PI_DeepONet class with th e KAN model
model= PI_DeepONet(model1,model2,model3)
model.to(device)
#     # Train the PI_DeepONet model
model.train(u_i1,u_i2, x_i, t_i,outputs_i,u_b1,u_b2,x_b, t_b, outputs_b)
data=pd.read_csv('data.csv')
x_test=data.iloc[:,1]
n=len(x_test)
t_test=data.iloc[:,2]
x_test=torch.tensor(x_test).float()
x_test=x_test.unsqueeze(1).to(device)
t_test=torch.tensor(t_test).float()
t_test=t_test.unsqueeze(1).to(device)

u_test1=(u1.view(1, -1)).repeat(n, 1).float().to(device)
u_test2 =(u2[0].view(1, -1)).repeat(n, 1).float().to(device)
x_test=min_max_normalize(x_test,x_bcs_min_value, x_bcs_max_value)
t_test=min_max_normalize(t_test,t_bcs_min_value,t_bcs_max_value)

s_pred = model.operator_net(u_test1,u_test2 ,x_test,t_test)
s_pred=s_pred*s_bcs_max_value
s_true=data.iloc[:,3]
s_true=torch.tensor(s_true)
error_s =(s_pred- s_true)/s_true
print('s_pred:\n',s_pred)
print('s_true:\n',s_true)
print('error_s:\n',error_s)
end_time=time.time()
rap_time=end_time-start_time
print('run-time:{}'.format(rap_time))