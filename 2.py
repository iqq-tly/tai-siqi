import numpy as nnp
import jax
import jax.numpy as np
from jax import random
# def calculate_V(T,r,v,M,K):
#     delta_T=T/M
#     S_max=100*K
#     delta_S= S_max/M
#
#     def get_call_matrix(M):
#         # call的三个边界条件
#         # 生成(M+1)*(M+1)的矩阵
#         f_matrx = nnp.matrix(nnp.array([0.0] * (M + 1) * (M + 1)).reshape((M + 1, M + 1)))
#         # 边界条件① S=0的时候，call=0
#         f_matrx[:, 0] = 0.0
#         # 边界条件②：在到期的时候，期权=max(δS*j-K, 0)
#         for i in range(M + 1):
#             f_matrx[M, i] = float(max(delta_S * i - K, 0))
#         # 边界条件③：S=S_max的时候，call=S_max-K
#         f_matrx[:, M] = float(S_max - K)
#         print("f_matrix shape : ", f_matrx.shape)
#         return f_matrx
#     def calculate_coeff(j):
#         vj2 = (v * j)**2
#         aj = 0.5 * delta_T * (r * j - vj2)
#         bj = 1 + delta_T * (vj2 + r)
#         cj = -0.5 * delta_T * (r * j + vj2)
#         return aj, bj, cj
#
#     def get_coeff_matrix(M):
#         #计算系数矩阵B
#         matrx = nnp.matrix(nnp.array([0.0]*(M-1)*(M-1)).reshape((M-1, M-1)))
#         a1, b1, c1 = calculate_coeff(1)
#         am_1, bm_1, cm_1 = calculate_coeff(M - 1)
#         matrx[0,0] = b1
#         matrx[0,1] = c1
#         matrx[M-2, M-3] = am_1
#         matrx[M-2, M-2] = bm_1
#         for i in range(2, M-1):
#             a, b, c = calculate_coeff(i)
#             matrx[i-1, i-2] = a
#             matrx[i-1, i-1] = b
#             matrx[i-1, i] = c
#         print("coeff matrix shape : ",  matrx.shape)
#         return matrx
#
#
#     f_matrx = get_call_matrix(M)
#     matrx = get_coeff_matrix(M)
#     inverse_m = matrx.I
#     for i in range(M, 0, -1):
#         # 迭代
#         Fi = f_matrx[i, 1:M]
#         Fi_1 = inverse_m * Fi.reshape((M-1, 1))
#         Fi_1 = list(nnp.array(Fi_1.reshape(1, M-1))[0])
#         f_matrx[i-1, 1:M]=Fi_1
#     return f_matrx
# m =100 # number of input samples
# N = 100  # number of input sensors
# P = 900  # number of output sensors, 100 for each side
# Q = 300 # number of collocation points for each input samp
# M = 1000
# r = 0.02023
# v= 0.142056898
# T=240
# K=2.411
# key = random.PRNGKey(0)
# subkeys = random.split(key, 10)
# idx = random.randint(subkeys[0], (30 // 3, 2), 0, max(M, M))
# call = calculate_V(T, r, v, M, K)
# s_bcs4 = call[idx[:, 0], idx[:, 1]]
# print(s_bcs4)
# s_bc4 = (np.array(s_bcs4)).reshape(-1, 1)
# x_bc1 = np.zeros((15// 3, 1))
# x_1=np.vstack([s_bc4,x_bc1 ])
# print(x_1)
import pandas as pd
data=pd.read_csv('etf_50_day_20.csv')
for i in range(1):
    K=data.iloc[i,4]
    print(K)