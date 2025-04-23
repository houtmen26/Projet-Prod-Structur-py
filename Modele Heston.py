import numpy as np
import pandas as pd
import numpy.random as nrd
import random as rd
import matplotlib.pyplot as plt
from math import *
import scipy.stats as si
from scipy.optimize import minimize


# Objectif est de définir le modele d'Heston et de le calibrer

option_data = pd.read_csv("options.csv",sep=';')

print('voici ce quon chercher : ',option_data)

print("voila la vol implicite : ",option_data['implied_volatility'])

option_data['Maturity'] = (pd.to_datetime(option_data['expiration']) - pd.to_datetime(option_data['price_date'])).dt.days/365.0

print("voici la maturité : ",option_data['Maturity'])

def payoff_heston(r,T, K, S0, rho, theta, k, neta, N, v0):
  delta_t = T/N
  S = [0] * N # Crée une liste de taille N pour S
  v = [0] * N # Crée une liste de taille N pour la vol sto
  vol = [0] * N
  S[0] = S0
  v[0] = v0
  vol[0] = sqrt(v0)
  for i in range(1,N):
    X = nrd.randn() # On simule une premiere gaussienne
    X1 = nrd.randn()  # On simule une seconde gaussienne
    S[i]  = S[i-1]*exp((r-0.5*v[i-1])*delta_t + sqrt(v[i-1])*(rho*sqrt(delta_t)*X+ sqrt(1-rho**2)*sqrt(delta_t)*X1))
    v[i] = v[i-1] + k*(theta - v[i-1])*delta_t + neta*sqrt(v[i-1])*sqrt(delta_t)*X + neta**2/4*delta_t*(X**2 - 1 )
    vol[i] = sqrt(v[i])
  return S, v, vol

def heston_option_price(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=10000, N=100, option_type='call'):

  total_payoff = 0

  for _ in range(n_simulations):
    # Générer une trajectoire
    S, _, _ = payoff_heston(r, T, K, S0, rho, theta, k, eta, N, v0)

    # Prix final de l'actif
    S_T = S[-1]

    # Calculer le payoff
    if option_type.lower() == 'call':
      payoff = max(S_T - K, 0)
    else:  # put
      payoff = max(K - S_T, 0)

    total_payoff += payoff

  # Prix moyen actualisé
  option_price = exp(-r * T) * (total_payoff / n_simulations)
  return option_price

def objective_function(params, option_data, S0, r):

  v0, rho, theta, k, eta = params

  total_error = 0
  for _, row in option_data.iterrows():
    K = row['strike']
    T = row['Maturity']
    market_price = row['mark']
    option_type = row['type']  # 'call' ou 'put'

    # Prix selon le modèle
    model_price = heston_option_price(r, T, K, S0, rho, theta, k, eta, v0,
                                      n_simulations=1000, N=100, option_type=option_type)

    # Erreur quadratique
    error = (model_price - market_price) ** 2
    total_error += error

  return total_error

def calibrate_heston_simplified(option_data, S0, r):
  # Fixer k et eta à des valeurs typiques
  k_fixed = 2.0
  eta_fixed = 0.3

  # Valeurs initiales pour v0, theta, rho
  initial_params = [0.04, -0.7, 0.04]

  # Bornes pour ces paramètres
  bounds = [(0.001, 0.25),  # v0
            (-0.99, 0.99),  # rho
            (0.001, 0.25)]  # theta

  def obj_simplified(params):
    v0, rho, theta = params
    return objective_function([v0, rho, theta, k_fixed, eta_fixed], option_data, S0, r)

  # Optimisation
  result = minimize(obj_simplified, initial_params, bounds=bounds, method='L-BFGS-B')

  v0, rho, theta = result.x
  return [v0, rho, theta, k_fixed, eta_fixed]

# Prix spot et taux sans risque
S0 = 100.0  # À ajuster selon votre actif
r = 0.01    # À ajuster selon le taux actuel

# Calibr ation
optimal_params = calibrate_heston_simplified(option_data, S0, r)
v0, rho, theta, k, eta = optimal_params

print(f"Paramètres calibrés:")
print(f"v0 = {v0:.6f} (variance initiale)")
print(f"rho = {rho:.6f} (corrélation)")
print(f"theta = {theta:.6f} (variance long-terme)")
print(f"k = {k:.6f} (vitesse de retour)")
print(f"eta = {eta:.6f} (vol de vol)")













