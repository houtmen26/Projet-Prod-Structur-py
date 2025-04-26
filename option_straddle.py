from Call_Heston import heston_option_Call,heston_option_PUT
from math import *
import pandas as pd
import numpy as np


def option_call_spread(r, T, K1,K2, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42)
    return heston_option_Call(r, T, K1, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42) - heston_option_Call(r, T, K2, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42)

def stradlle(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42):
    return heston_option_Call(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42) + heston_option_PUT(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42)

def strangle(r, T, K1,K2, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42):
    return heston_option_Call(r, T, K2, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42) + heston_option_PUT(r, T, K1, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42)

def strip(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42):
    return heston_option_Call(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42) + 2*heston_option_PUT(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42)

def strap(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42):
    return 2*heston_option_Call(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42) + heston_option_PUT(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42)
