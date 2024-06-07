import jax
import torch
import jax.numpy as np
import numpy as nnp
from jax import random, grad, vmap, jit, hessian
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
import matplotlib.pyplot as plt
import pandas as pd
import time



def calculate_V(T,r,v,M,K):
    delta_T=T/M
    S_max=3*K
    delta_S= S_max/M

    def get_call_matrix(M):
        f_matrx = nnp.matrix(nnp.array([0.0] * (M + 1) * (M + 1)).reshape((M + 1, M + 1)))
        f_matrx[:, 0] = 0.0
        for i in range(M + 1):
            f_matrx[M, i] = float(max(delta_S * i - K, 0))
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
        Fi = f_matrx[i, 1:M]
        Fi_1 = inverse_m * Fi.reshape((M-1, 1))
        Fi_1 = list(nnp.array(Fi_1.reshape(1, M-1))[0])
        f_matrx[i-1, 1:M]=Fi_1
    return f_matrx

key = random.PRNGKey(1)
subkeys = random.split(key, 10)
Q = 100 # number of collocation points for each input samp
M = 8000
r =0.025610
v=0.165856529
T=1
K=2.411
x_max = 3* K
delta_T = T / M
delta_S = x_max / M


idx = random.randint(subkeys[1], (100,2), 0, max(M,M))
call = calculate_V(T,r,v,M,K)
call=np.asarray(call)
data=pd.read_csv('data.csv')
x_test=data.iloc[:,1]
t_test=data.iloc[:,2]
s_true=data.iloc[:,3]


x_bc4 = list(x_test/delta_S)
x_bc4 = [round(num) for num in x_bc4]
t_bc4 = list((T -t_test/365) / delta_T)
t_bc4 = [round(num) for num in t_bc4]
s_pred= call[t_bc4,x_bc4]
error_s =(s_pred- s_true)/s_true
print('s_pred',s_pred)
print('s_true',s_true)
print(error_s)