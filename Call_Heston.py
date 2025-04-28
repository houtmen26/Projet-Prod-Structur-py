
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import numpy.random as nrd
import random as rd
import matplotlib.pyplot as plt
from math import *
import scipy.stats as si
from scipy.optimize import minimize
import time
from Modele_Heston import payoff_heston


S0 = 215  # À ajuster selon votre actif
r = 0.045  # À ajuster selon le taux actuel

### L'objectif ici  étant de pricer un call avec le modele de Heston

# On commence par recuperer les parametre calibré par notre modele puis on va simuler S et V en meme temps.

# Une fois cela effectué on pourra par MC calculter S_T donc S[-1] et farie (S_T-K) et faire la moyenne dess simulation,
# il faudra faire des test pour que ca dure un peu moins de 1 minutes sinon ca sera trop long (call spread, strip strap ect prendront
# trop de temps)

param = [0.098728,-0.489983,0.017104, 0.180130,0.220737]

def heston_option_price_call(r, T, K, S0, rho, theta, k, eta, v0, Nmc=1000, N=100, seed=None):
  if seed is None:
    seed = np.random.randint(0, 10000)
  S_path = payoff_heston(r,T, K, S0, rho, theta, k, eta, N, Nmc,seed,v0)
  S_T = S_path[:,-1]
  payoff = np.maximum(S_T - K,0)
  return exp(-r*T)*np.mean(payoff)

def heston_option_price_put(r, T, K, S0, rho, theta, k, eta, v0, Nmc=1000, N=100,  seed=None):
  if seed is None:
    seed = np.random.randint(0, 10000)
  S_path = payoff_heston(r,T, K, S0, rho, theta, k, eta, N, Nmc,seed,v0)
  S_T = S_path[:,-1]
  payoff = np.maximum(K - S_T,0)
  return exp(-r*T)*np.mean(payoff)
