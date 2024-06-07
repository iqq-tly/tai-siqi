import numpy as np
from scipy.stats import norm


def black_scholes(S, K, T, r, sigma, option_type='call'):

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    return option_price



S =2.4260
K =2.411
T = 20/365
r = 0.165856529
sigma = 0.025610


call_price = black_scholes(S, K, T, r, sigma, option_type='call')
print(f"call_price: {call_price}")