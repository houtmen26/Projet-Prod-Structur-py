from Call_Heston import heston_option_price_put,heston_option_price_call
from math import *
import pandas as pd
import numpy as np

S0 = 215  # À ajuster selon votre actif
r = 0.045  # À ajuster selon le taux actuel
param = [0.098728,-0.489983,0.017104, 0.180130,0.220737]

def option_call_spread(r, T, K1,K2, S0, rho, theta, k, eta, v0, Nmc, N):
    return heston_option_price_call(r, T, K1, S0, rho, theta, k, eta, v0, Nmc, N, option_type=None, seed=None) -  heston_option_price_call(r, T, K2, S0, rho, theta, k, eta, v0, Nmc, N, option_type=None, seed=None)

def stradlle(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N):
    return heston_option_price_call(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N, option_type=None, seed=None) + heston_option_price_put(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N, option_type=None, seed=None)

def strangle(r, T, K1,K2, S0, rho, theta, k, eta, v0, Nmc, N):
    return heston_option_price_call(r, T, K1, S0, rho, theta, k, eta, v0, Nmc, N, option_type=None, seed=None) +  heston_option_price_call(r, T, K2, S0, rho, theta, k, eta, v0, Nmc, N, option_type=None, seed=None)

def strip(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N):
    return heston_option_price_call(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N, option_type=None, seed=None) + 2*heston_option_price_put(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N, option_type=None, seed=None)

def strap(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N):
    return 2*heston_option_price_call(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N, option_type=None, seed=None) + heston_option_price_put(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N, option_type=None, seed=None)

def option_call_binaire(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N,h1):
    # Dans la littérature il est dit qu'une bonne valeur de H1 est 0,01% du strike
    return heston_option_price_call(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N, option_type=None, seed=None) -  heston_option_price_call(r, T, K+h1, S0, rho, theta, k, eta, v0, Nmc, N, option_type=None, seed=None)
