import numpy as np
import pandas as pd
import numpy.random as nrd
import random as rd
import matplotlib.pyplot as plt
from math import *
import scipy.stats as si

# Objectif est de définir le modele d'Heston et de le calibrer

option = pd.read_csv("options.csv")
print(option)

def payoff_heston(r,T, K, S0, rho, theta, k, neta, N, v0):
  delta_t = T/N
  S = [0] * N
  v = [0] * N
  vol = [0] * N
  S[0] = S0
  v[0] = v0
  vol[0] = sqrt(v0)
  for i in range(1,N):
    X = nrd.randn()
    X1 = nrd.randn()
    S[i]  = S[i-1]*exp((r-0.5*v[i-1])*delta_t + sqrt(v[i-1])*(rho*sqrt(delta_t)*X+ sqrt(1-rho**2)*sqrt(delta_t)*X1))
    v[i] = v[i-1] + k*(theta - v[i-1])*delta_t + neta*sqrt(v[i-1])*sqrt(delta_t)*X + neta**2/4*delta_t*(X**2 - 1 )
    vol[i] = sqrt(v[i])
  return S, v, vol

