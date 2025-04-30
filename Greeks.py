from Call_Heston import heston_option_price_put,heston_option_price_call
from math import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from produit_financier import *


def calcul_delta(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N,h1): #
    delta = heston_option_price_call(r, T, K, S0 +h1, rho, theta, k, eta, v0, Nmc, N, seed=None) - heston_option_price_call(r, T, K, S0 -h1, rho, theta, k, eta, v0, Nmc, N,  seed=None)
    return delta/(2*h1)

def calcul_theta(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N,h1):
    theta = heston_option_price_call(r, T+h1, K, S0 , rho, theta, k, eta, v0, Nmc, N,  seed=None) - heston_option_price_call(r, T-h1, K, S0, rho, theta, k, eta, v0, Nmc, N,  seed=None)
    return theta/(2*h1)

def calcul_Vega(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N,h1):
    Vega= heston_option_price_call(r, T, K, S0 , rho, theta, k, eta, v0+h1, Nmc, N, seed=None) - heston_option_price_call(r, T, K, S0, rho, theta, k, eta, v0-h1, Nmc, N,  seed=None)
    return Vega/(2*h1)

def calcul_Gamma(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N,h1):
    price_plus = heston_option_price_call(r, T, K, S0 + h1, rho, theta, k, eta, v0, Nmc, N,  seed=None)
    price_center = heston_option_price_call(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N,  seed=None)
    price_minus = heston_option_price_call(r, T, K, S0 - h1, rho, theta, k, eta, v0, Nmc, N, seed=None)
    Gamma = (price_plus - 2 * price_center + price_minus) / (h1 ** 2)
    return Gamma

def calcul_rho(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N,h1):
    rho = heston_option_price_call(r+h1, T, K, S0 , rho, theta, k, eta, v0, Nmc, N,  seed=None) - heston_option_price_call(r-h1, T, K, S0, rho, theta, k, eta, v0-h1, Nmc, N,  seed=None)
    return rho/(2*h1)


