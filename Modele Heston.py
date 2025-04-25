import numpy as np
import pandas as pd
import numpy.random as nrd
import random as rd
import matplotlib.pyplot as plt
from math import *
import scipy.stats as si
from scipy.optimize import minimize
import time

# Objectif est de définir le modele d'Heston et de le calibrer

option_data = pd.read_csv("options.csv",sep=';')

print('voici ce quon chercher : ',option_data)

print("voila la vol implicite : ",option_data['implied_volatility'])

option_data['Maturity'] = (pd.to_datetime(option_data['expiration']) - pd.to_datetime(option_data['price_date'])).dt.days/365.0

print("voici la maturité : ",option_data['Maturity'])
# Définir une plage plus large de strikes qui couvre mieux votre espace
strikes_ranges = [
  (50, 75),  # Deep ITM
  (75, 95),  # ITM
  (95, 105),  # ATM
  (105, 150),  # OTM
  (150, 300)  # Deep OTM
]

maturity_ranges = [
  (0.05, 0.15),  # Court terme
  (0.15, 0.5),  # Moyen terme
  (0.5, 1.0) , # Long terme (si disponible)
  (1,1.5), # Long terme (si disponible)
  (1.5,2.5) # Long terme (si disponible)
]

selected_options = []

# Sélectionner les options de manière stratifiée
for s_range in strikes_ranges:
  for m_range in maturity_ranges:
    bucket_options = option_data[
      (option_data['strike'] >= s_range[0]) &
      (option_data['strike'] <= s_range[1]) &
      (option_data['Maturity'] > m_range[0]) &
      (option_data['Maturity'] <= m_range[1])
      ]

    if len(bucket_options) > 0:
      # Prendre 1-2 options de chaque bucket
      selected_options.append(bucket_options.iloc[0:min(2, len(bucket_options))])

# Concaténer les résultats
option_data = pd.concat(selected_options)


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
    sqrt_v = sqrt(v[i - 1]) if v[i - 1] > 0 else 0
    S[i]  = S[i-1]*exp((r-0.5*v[i-1])*delta_t + sqrt_v*(rho*sqrt(delta_t)*X+ sqrt(1-rho**2)*sqrt(delta_t)*X1))
    v[i] = v[i-1] + k*(theta - v[i-1])*delta_t + neta*sqrt_v*sqrt(delta_t)*X + neta**2/4*delta_t*(X**2 - 1 )
    vol[i] = sqrt(abs(v[i]))
  return S, v, vol

def heston_option_price(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call'):

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
  for i, row in enumerate(option_data.itertuples(), start=1):
    K = row.strike
    T = row.Maturity
    market_price = row.mark
    option_type = row.type

    if pd.isna(market_price):
      continue

    print(f"Calibration sur option {i}/{len(option_data)}")

    # Prix selon le modèle
    model_price = heston_option_price(r, T, K, S0, rho, theta, k, eta, v0,
                                      n_simulations=1000, N=100, option_type=option_type)

    # Erreur quadratique
    error = (model_price - market_price) ** 2
    total_error += error

  return total_error

def calibrate_heston_simplified(option_data, S0, r):
  # Fixer k et eta à des valeurs typiques


  # Valeurs initiales pour v0, theta, rho,k,neta
  initial_params = [0.04, -0.7, 0.04,1,0.5]

  # Bornes pour ces paramètres
  bounds = [(0.001, 2),  # v0 (vol de départ)
            (-0.99, 0.99),  # rho (correl donc entre -1 et 1)
            (0.001, 2),  # theta
            (0.01, 20), # k (retour a la moyenne)
            (0.01, 7)]  # neta (vol of vol)


  def obj_simplified(params):
    v0, rho, theta,k,neta = params
    return objective_function([v0, rho, theta, k, neta], option_data, S0, r)

  # Optimisation
  result = minimize(obj_simplified, initial_params, bounds=bounds, method='L-BFGS-B')
  if not result.success:
    print(" Optimisation échouée :", result.message)

  v0, rho, theta,k,neta = result.x
  return [v0, rho, theta, k, neta]

# Prix spot et taux sans risque
S0 = 100.0  # À ajuster selon votre actif
r = 0.01    # À ajuster selon le taux actuel

# Calibration
start = time.time()
optimal_params = calibrate_heston_simplified(option_data, S0, r)
v0, rho, theta, k, eta = optimal_params

end = time.time()
print(f"\n⏱️ Temps total de calibration : {end - start:.2f} secondes\n")

print(f"Paramètres calibrés:")
print(f"v0 = {v0:.6f} (variance initiale)")
print(f"rho = {rho:.6f} (corrélation)")
print(f"theta = {theta:.6f} (variance long-terme)")
print(f"k = {k:.6f} (vitesse de retour)")
print(f"eta = {eta:.6f} (vol de vol)")













