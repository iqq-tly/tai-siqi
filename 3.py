import jax
import torch
import jax.numpy as np
import numpy as nnp
from jax import random, grad, vmap, jit, hessian
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from jax.nn import relu, elu,leaky_relu
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
import matplotlib.pyplot as plt
import pandas as pd
import time
start_time=time.time()

from scipy.interpolate import griddata
# Data generator
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s,
                 batch_size=128, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s

        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx, :]
        y = self.y[idx, :]
        u = self.u[idx, :]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs

# Define modified MLP
def modified_MLP(layers, activation=relu):
  def xavier_init(key, d_in, d_out):
      glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
      W = glorot_stddev * random.normal(key, (d_in, d_out))
      b = np.zeros(d_out)
      return W, b

  def init(rng_key):
      U1, b1 =  xavier_init(random.PRNGKey(12345), layers[0], layers[1])
      U2, b2 =  xavier_init(random.PRNGKey(54321), layers[0], layers[1])
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          W, b = xavier_init(k1, d_in, d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return (params, U1, b1, U2, b2)

  def apply(params, inputs):
      params, U1, b1, U2, b2 = params
      U = activation(np.dot(inputs, U1) + b1)
      V = activation(np.dot(inputs, U2) + b2)
      for W, b in params[:-1]:
          outputs = activation(np.dot(inputs, W) + b)
          inputs = np.multiply(outputs, U) + np.multiply(1 - outputs, V)
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply

# Define the model
class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers):
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply =modified_MLP(branch_layers, activation=relu)
        self.trunk_init, self.trunk_apply =modified_MLP(trunk_layers, activation=relu )

        # Initialize
        branch_params = self.branch_init(rng_key = random.PRNGKey(1))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(4))
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(0.001 ,
                                                                      decay_steps=2000,
                                                                      decay_rate=0.9))
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # Loggers
        self.loss_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    # Define DeepONet architecture
    def operator_net(self, params, u, x, t):
        branch_params, trunk_params = params
        y = np.stack([x, t])
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = np.sum(B * T)
        return outputs

    # Define PDE residual
    def residual_net(self, params, u, x, t):
        s = self.operator_net(params, u, x, t)
        s_x = grad(self.operator_net, argnums=2)(params, u, x, t)
        s_xx = grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, x, t)
        s_t = grad(self.operator_net, argnums=3)(params, u, x, t)
        res =s_t-(0.5)*(0.142056898**2)*(x**2)*s_xx-0.02023*x*s_x+0.02023*s
        return res

    # Define boundary loss
    def loss_bcs(self, params, batch):
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:, 0], y[:, 1])
        # Compute loss
        loss = np.mean((outputs.flatten() - s_pred) ** 2)
        return loss

    # Define residual loss
    def loss_res(self, params, batch):
      # Fetch data
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.residual_net, (None, 0, 0, 0))(params, u, y[:, 0], y[:, 1])

        loss = np.mean((outputs.flatten() - pred) ** 2)
        return loss

    # Define total loss

    def loss(self, params, bcs_batch, res_batch):
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss =loss_bcs +loss_res
        return loss

    # Define a compiled update step

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, bcs_dataset, res_dataset, nIter=10000):
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            bcs_batch = next(bcs_data)
            res_batch = next(res_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, bcs_batch, res_batch)

            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, bcs_batch, res_batch)
                loss_bcs_value = self.loss_bcs(params, bcs_batch)
                loss_res_value = self.loss_res(params, res_batch)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                # Print losses
                pbar.set_postfix({'Loss': loss_value,
                                      'loss_bcs': loss_bcs_value,
                                      'loss_physics': loss_res_value})

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = self.operator_net(params, U_star, Y_star[0], Y_star[1])
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        r_pred =self.residual_net(params, U_star, Y_star[0], Y_star[1])
        return r_pred


# Use double precision to generate data (due to GP sampling)
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)

# Deinfe initial and boundary conditions for advection equation
def f1(x,k):
  return np.where(x==0,0,1)
def f2(x,k):
  return np.where(x == 100*k,x-k, 0)
def g(x,t,k):
  return np.where(t==0,np.max(x-k),0)


#归一化
def min_max_normalize(x, min_val, max_val):
    normalized_x = (x - min_val) / (max_val - min_val)
    return normalized_x
def calculate_V(T,r,v,M,K):
    delta_T=T/M
    S_max=100*K
    delta_S= S_max/M

    def get_call_matrix(M):
        # call的三个边界条件
        # 生成(M+1)*(M+1)的矩阵
        f_matrx = nnp.matrix(nnp.array([0.0] * (M + 1) * (M + 1)).reshape((M + 1, M + 1)))
        # 边界条件① S=0的时候，call=0
        f_matrx[:, 0] = 0.0
        # 边界条件②：在到期的时候，期权=max(δS*j-K, 0)
        for i in range(M + 1):
            f_matrx[M, i] = float(max(delta_S * i - K, 0))
        # 边界条件③：S=S_max的时候，call=S_max-K
        f_matrx[:, M] = float(S_max - K)
        print("f_matrix shape : ", f_matrx.shape)
        return f_matrx
    def calculate_coeff(j):
        vj2 = (v * j)**2
        aj = 0.5 * delta_T * (r * j - vj2)
        bj = 1 + delta_T * (vj2 + r)
        cj = -0.5 * delta_T * (r * j + vj2)
        return aj, bj, cj

    def get_coeff_matrix(M):
        #计算系数矩阵B
        matrx = nnp.matrix(nnp.array([0.0]*(M-1)*(M-1)).reshape((M-1, M-1)))
        a1, b1, c1 = calculate_coeff(1)
        am_1, bm_1, cm_1 = calculate_coeff(M - 1)
        matrx[0,0] = b1
        matrx[0,1] = c1
        matrx[M-2, M-3] = am_1
        matrx[M-2, M-2] = bm_1
        for i in range(2, M-1):
            a, b, c = calculate_coeff(i)
            matrx[i-1, i-2] = a
            matrx[i-1, i-1] = b
            matrx[i-1, i] = c
        print("coeff matrix shape : ",  matrx.shape)
        return matrx


    f_matrx = get_call_matrix(M)
    matrx = get_coeff_matrix(M)
    inverse_m = matrx.I
    for i in range(M, 0, -1):
        # 迭代
        Fi = f_matrx[i, 1:M]
        Fi_1 = inverse_m * Fi.reshape((M-1, 1))
        Fi_1 = list(nnp.array(Fi_1.reshape(1, M-1))[0])
        f_matrx[i-1, 1:M]=Fi_1
    return f_matrx
# Geneate training data corresponding to one input sample
def generate_one_training_data(key,M,r,T,v,m,P,Q,K):
    subkeys = random.split(key, 10)
    length_scale=0.2
    x_max = 100 * K
    delta_T = T/ M
    delta_S = x_max/M
    # Generate a GP sample
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = np.linspace(0,x_max, N)[:, None]
    KK = RBF(X, X, gp_params)
    L = np.linalg.cholesky(KK + jitter * np.eye(N))
    gp_sample = np.dot(L, random.normal(subkeys[0], (N,)))
    gp_sample = np.abs(gp_sample)
    v_fn = lambda x: np.interp(x, X.flatten(), gp_sample)
    xx = np.linspace(0,x_max,m)
    u=v_fn(xx)
    idx = random.randint(subkeys[1], (1000,2), 0, max(M,M))
    call = calculate_V(T,r,v,M,K)
    call=np.asarray(call)
    s_bcs4= call[idx[:,1], idx[:,0]]
    s_bc4=s_bcs4.reshape(-1, 1)
    x_bc4 = (idx[:, 0] * delta_S).reshape(-1, 1)
    t_bc4 = (T - idx[:, 1] * delta_T).reshape(-1, 1)

    np_K=K*(np.ones((P // 3, 1)))

    x_bc1 = np.zeros((P // 3, 1))
    x_bc2 = 100*K*(np.ones((P //3, 1)))
    x_bc3 = random.uniform(subkeys[2], shape=(P // 3, 1), minval=0, maxval= x_max)
    x_bcs = np.vstack([x_bc1, x_bc2, x_bc3,x_bc4])
    x_bcs= min_max_normalize(x_bcs,0,x_max)

    t_bc1 = random.uniform(subkeys[3], shape=(P // 3, 1), minval=0, maxval=T)
    t_bc2 = random.uniform(subkeys[3], shape=(P // 3, 1), minval=0, maxval=T)
    t_bc3 = np.zeros((P // 3, 1))
    t_bcs = np.vstack([t_bc1, t_bc2,t_bc3,t_bc4])
    t_bcs = min_max_normalize(t_bcs, 0, T)

    u_train = np.tile(u, (P+1000, 1))
    y_train = np.hstack([x_bcs, t_bcs])
    s_bc1=vmap(f1,(0,0))(x_bc1,np_K)
    s_bc1 =np.array(list(s_bc1))
    s_bc1 =s_bc1.reshape(-1,1)
    s_bc2 =vmap(f2,(0,0))(x_bc2,np_K)
    s_bc2 = np.array(list(s_bc2))
    s_bc3 = vmap(g,(0,0,0))(x_bc3,t_bc3, np_K)
    s_bc3 = np.array(list(s_bc3))
    s_bc3 = s_bc3.reshape(-1, 1)
    s_train = np.vstack([s_bc1,s_bc2,s_bc3,s_bc4])
    s_min_value = np.min(s_train)
    s_max_value = np.max(s_train)
    s_train= min_max_normalize(s_train, s_min_value, s_max_value)
    x_r_1 = random.uniform(subkeys[4], shape=(Q, 1), minval=0, maxval= x_max)
    t_r_1 = random.uniform(subkeys[5], shape=(Q, 1), minval=0, maxval=T)
    x_r_1 = min_max_normalize(x_r_1, 0, x_max)
    t_r_1 = min_max_normalize(t_r_1, 0, T)
    x_r=np.vstack([x_bcs,x_r_1])
    t_r=np.vstack([t_bcs,t_r_1])
    u_r_train = np.tile(u, (P+1000+Q, 1))
    y_r_train = np.hstack([x_r, t_r])
    s_r_train = np.zeros((P+1000+Q, 1))

    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train,s_max_value
# Geneate training data corresponding to N input sample
def generate_training_data(key,M,r,T,V,m,P,Q,K,N):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    u_train, y_train, s_train, u_r_train, y_r_train, s_r_train,s_max_value= \
        vmap(generate_one_training_data, (0, None, None,None,None, None, None,None,None))(keys,M,r,T,V,m,P,Q,K)
    u_train = np.float32(u_train.reshape(N * (P+1000), -1))
    y_train = np.float32(y_train.reshape(N * (P+1000), -1))
    s_train = np.float32(s_train.reshape(N * (P+1000), -1))

    u_r_train = np.float32(u_r_train.reshape(N * (P+1000+Q), -1))
    y_r_train = np.float32(y_r_train.reshape(N * (P+1000+Q), -1))
    s_r_train = np.float32(s_r_train.reshape(N * (P+1000+Q), -1))

    config.update("jax_enable_x64", False)
    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train,s_max_value



###训练模型
key = random.PRNGKey(1)
m =100 # number of input samples
N = 600  # number of input sensors
P = 150  # number of output sensors, 100 for each side
Q = 100 # number of collocation points for each input samp
M = 5000
r = 0.02023
v= 0.142056898
T=240/365
#test prepation
config.update("jax_enable_x64", True)
data=pd.read_csv('etf_50_day_20.csv')
for i in range(1,2):
    K=data.iloc[i,4]
    print(K)
    u_bcs_train, y_bcs_train, s_bcs_train, u_res_train, y_res_train, s_res_train,s_max_value = \
        generate_training_data(key, M, r, T, v, m, P,Q,K,N)
    u_test = u_bcs_train[0]
    s_test=data.iloc[i,5]
    x=2.46
    t=(data.iloc[i,6])/(365*T)
    x=min_max_normalize(x,0,100*K)
    y_test=np.hstack([x, t])
    print(y_test)
    u_test = np.float32(u_test)
    y_test = np.float32(y_test)
#train
    batch_size = 1000
    bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
    res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size)
    # Initialize model
    branch_layers = [m,100,100,100,100,100,100]
    trunk_layers = [2,100,100,100,100,100,100]

    model = PI_DeepONet(branch_layers, trunk_layers)
    model.train(bcs_dataset, res_dataset, nIter=2000)
    params = model.get_params(model.opt_state)
    s_pred = model.predict_s(params,u_test, y_test)
    print(s_pred)
    print(s_max_value[0])
    s_pred=s_pred*s_max_value[0]
    print(s_pred)



####for 循环
# list_s_ture=[]
# list_s_pred=[]
# error=[]
# ###实战
# data=pd.read_csv('etf_50_day_20.csv')
# for i in range(1):
#     K=data.iloc[i,4]
#     s_test=data.iloc[i,5]
#     x=2.46
#     t=data.iloc[i,6]

    # error_s = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test)
    # error.append(error_s)
    # list_s_ture.append(s_test)
    # list_s_ture.append(s_pred)
#Plot for loss function
# print(error)
# print(list_s_ture)
# print(list_s_pred)
plt.figure(figsize = (6,5))
# plt.plot(s_test , lw=2, label='test')
# plt.plot(s_pred, lw=2, label='pred')



# def compute_error(s_test,s_pred):
#     # Compute relative l2 error
#     error_s = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test)
#     return error_s

# print(k)
# data=pd.read_csv('etf_50_day_20.csv')
# k=np.array(list(data.iloc[:5,4]))
# k.reshape(-1,1)
# s=np.array(list(data.iloc[:5,5]))
# s.reshape(-1,1)
# t=np.array(list(data.iloc[:5,6]))
# t.reshape(-1,1)
# x=2.46*(np.ones((len(k),1)))
# key = random.PRNGKey(0)
# s_pred=vmap(model_test,(None,0,0,0))(key,k,x,t)
# error_s = vmap(compute_error,(0,0))(s,s_pred)
# print(error_s )
# plt.figure(figsize = (6,5))
# plt.plot(error_s,lw=2,label='error')
# plow.show()






end_time=time.time()
rap_time=end_time-start_time
print('运行时间为：{}'.format(rap_time))




