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

print("\n✅ Voici les options sélectionnées pour la calibration :")
print(option_data[['strike', 'Maturity', 'mark', 'type']])

def payoff_heston_vectorized(r, T, S0, rho, theta, k, eta, v0, N=100, n_paths=1000, seed=42):
    np.random.seed(seed)
    delta_t = T / N
    S = np.full((n_paths, N), S0)
    v = np.full((n_paths, N), v0)
    sqrt_dt = sqrt(delta_t)

    for i in range(1, N):
        Z1 = np.random.randn(n_paths)
        Z2 = np.random.randn(n_paths)
        sqrt_v = np.sqrt(np.maximum(v[:, i-1], 0))
        S[:, i] = S[:, i-1] * np.exp((r - 0.5 * v[:, i-1]) * delta_t + sqrt_v * (rho * sqrt_dt * Z1 + sqrt(1 - rho**2) * sqrt_dt * Z2))
        v[:, i] = np.maximum(
            v[:, i-1] + k * (theta - v[:, i-1]) * delta_t + eta * sqrt_v * sqrt_dt * Z1 + 0.25 * eta**2 * delta_t * (Z1**2 - 1),
            0
        )
    return S


def heston_option_price(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42):
    S_paths = payoff_heston_vectorized(r, T, S0, rho, theta, k, eta, v0, N=N, n_paths=n_simulations, seed=seed)
    S_T = S_paths[:, -1]
    if option_type.lower() == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    return exp(-r * T) * np.mean(payoffs)
# ==============================================
# Fonction objectif
def objective_function(params, option_data, S0, r):
    v0, rho, theta, k, eta = params
    total_error = 0
    constant_seed = 12345

    print("\n=== Démarrage de l'évaluation de la fonction objectif ===\n")

    for i, row in option_data.iterrows():
        K = row['strike']
        T = row['Maturity']
        market_price = row['mark']
        option_type = row['type']

        if pd.isna(market_price):
            continue

        print(f"Option {i+1}/{len(option_data)} : Strike={K}, T={T:.4f}, Type={option_type}, Market={market_price:.4f}")

        model_price = heston_option_price(r, T, K, S0, rho, theta, k, eta, v0,
                                          n_simulations=1000, N=100, option_type=option_type, seed=constant_seed)

        print(f"   ➔ Model price = {model_price:.4f}")

        denominator = max(0.01, abs(market_price))
        error = ((model_price - market_price) / denominator) ** 2

        weight = 1.0
        if abs(K / S0 - 1) < 0.05:  # ATM
            weight *= 2.0
        if T < 0.25:  # Court terme
            weight *= 1.5

        total_error += error * weight

    print(f"\nTotal erreur pour cet essai: {total_error:.6f}\n")
    return total_error

# ==============================================
# Calibration
def calibrate_heston(option_data, S0, r):
    bounds = [(0.001, 2), (-0.99, 0.99), (0.001, 2), (0.01, 20), (0.01, 7)]
    initial_guesses = [
        [0.04, -0.7, 0.04, 1.0, 0.5],
        [0.1, -0.5, 0.1, 2.0, 0.3],
        [0.05, -0.3, 0.05, 3.0, 0.4]
    ]

    best_result = None
    best_error = float('inf')

    for init in initial_guesses:
        print(f"🔍 Tentative de calibration à partir de {init}")
        result = minimize(lambda p: objective_function(p, option_data, S0, r),
                          init, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-8})
        print(f"Résultat intermédiaire : erreur={result.fun:.6f}")

        if result.fun < best_error:
            best_error = result.fun
            best_result = result

    v0, rho, theta, k, eta = best_result.x
    return v0, rho, theta, k, eta

# ==============================================
# Lancement
S0 = 215.0
r = 0.05

start = time.time()
v0, rho, theta, k, eta = calibrate_heston(option_data, S0, r)
end = time.time()

print(f"\n⏱️ Temps total de calibration : {end - start:.2f} secondes")
print(f"\n=== Paramètres calibrés ===")
print(f"v0 = {v0:.6f}")
print(f"rho = {rho:.6f}")
print(f"theta = {theta:.6f}")
print(f"k = {k:.6f}")
print(f"eta = {eta:.6f}")











def payoff_heston(r,T, K, S0, rho, theta, k, neta, N, Nmc,seed,v0):
  delta_t = T/N
  np.random.seed(seed)
  S = np.full((Nmc, N), S0)
  v = np.full((Nmc, N), v0)
  sqrt_dt = sqrt(delta_t)


  for i in range(1,N):
    X = np.random.randn(Nmc)  # On simule une premiere gaussienne
    X1 = np.random.randn(Nmc)  # On simule une seconde gaussienne
    sqrt_v = np.sqrt(np.maximum(v[:, i-1], 0))
    S[:,i]  = S[:,i-1]*np.exp((r-0.5*v[:,i-1])*delta_t + sqrt_v*(rho*sqrt(delta_t)*X+ sqrt(1-rho**2)*sqrt(delta_t)*X1))
    v[:,i] = np.maximum(0, v[:,i-1] + k*(theta - v[:,i-1])*delta_t + neta*sqrt_v*sqrt(delta_t)*X + 0.25*neta**2*delta_t*(X**2 - 1))

  return S






