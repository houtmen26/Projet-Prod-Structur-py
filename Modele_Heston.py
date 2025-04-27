import numpy as np
import pandas as pd
import numpy.random as nrd
import random as rd
import matplotlib.pyplot as plt
from math import *
import scipy.stats as si
from scipy.optimize import minimize
import time
from multiprocessing import Pool
from scipy.optimize import differential_evolution


# Objectif est de définir le modele d'Heston et de le calibrer

option_data = pd.read_csv("options.csv",sep=';')

print('voici ce quon chercher : ',option_data)

print("voila la vol implicite : ",option_data['implied_volatility'])

option_data['Maturity'] = (pd.to_datetime(option_data['expiration']) - pd.to_datetime(option_data['price_date'])).dt.days/365.0

print("voici la maturité : ",option_data['Maturity'])
# Définir une plage plus large de strikes qui couvre mieux votre espace
strikes_ranges = [
  (150,180),
  (180,200),
  (200,220),
  (220,250),
  (250,300),
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



def payoff_heston(r,T, K, S0, rho, theta, k, eta, N, Nmc,seed,v0):
  delta_t = T/N
  np.random.seed(seed)
  S = np.zeros((Nmc, N + 1))  # Changement: N+1 pour inclure S0
  v = np.zeros((Nmc, N + 1))  # Changement: N+1 pour inclure v0

  S[:, 0] = S0
  v[:, 0] = v0
  sqrt_dt = sqrt(delta_t)

  for i in range(1,N+1):
    Z1 = np.random.normal(size=Nmc)
    Z2 = np.random.normal(size=Nmc)
    Z2_corr = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
    sqrt_v = np.sqrt(np.maximum(v[:, i-1], 0))
    S[:, i] = S[:, i - 1] * np.exp((r - 0.5 * v[:, i - 1]) * delta_t + sqrt_v * sqrt_dt * Z2_corr)
    # Application du schéma d'Euler pour v (avec correction pour éviter les valeurs négatives)
    v[:, i] = np.maximum(0, v[:, i - 1] + k * (theta - v[:, i - 1]) * delta_t + eta * sqrt_v * sqrt_dt * Z1)

  return S

def heston_option_price(r, T, K, S0, rho, theta, k, eta, v0, Nmc=30000, N=100, option_type=None, seed=None):
  if seed is None:
    seed = np.random.randint(0, 10000)
  S_path = payoff_heston(r,T, K, S0, rho, theta, k, eta, N, Nmc,seed,v0)
  S_T = S_path[:,-1]
  if option_type.lower() == 'call':
    payoff = np.maximum(S_T - K,0)
  else :
    payoff = np.maximum(K-S_T,0)

  return exp(-r*T)*np.mean(payoff)


def objective_function(params, option_data, S0, r):
  v0, rho, theta, k, eta = params
  total_error = 0
  np.random.seed(12345)

  for i, row in enumerate(option_data.itertuples(), start=1):
    K = row.strike
    T = row.Maturity
    market_price = row.mark
    option_type = row.type

    if pd.isna(market_price):
      continue

    random_seed = np.random.randint(0, 1000000)
    # Prix selon le modèle - AUGMENTER le nombre de simulations et UTILISER le seed
    model_price = heston_option_price(r, T, K, S0, rho, theta, k, eta, v0, Nmc= 30000, N=100, option_type=option_type, seed=random_seed)

    # Pour le debug, afficher les prix
    print(f"  Strike={K}, T={T}, Market={market_price:.4f}, Model={model_price:.4f}")
    price_diff = model_price - market_price
    normalized_error = price_diff ** 2 / (market_price ** 2 + 1.0)

    moneyness = K / S0
    weight = 1.0
    if 0.95 <= moneyness <= 1.05:  # ATM
      weight *= 3.0
    if T < 0.25:  # Short-term
      weight *= 2.0

    total_error += normalized_error  * weight

  print(f"Erreur totale pour ces paramètres: {total_error:.6f}")

  return total_error

def calibrate_heston_simplified(option_data, S0, r):
    # Limiter éventuellement le nombre d'options pour les tests
  if len(option_data) > 10:
    print(f"Attention: nombreuses options ({len(option_data)}). Possible ralentissement.")
  print("\n=== Vérification du stock et des options ===\n")
  print(f"Prix spot utilisé (S0) : {S0}")
  print("\nAperçu des premières options :")
  print(option_data[['strike', 'mark', 'Maturity', 'type']].head(10))
  print("\nRésumé statistique des strikes et des prix de marché :")
  print(option_data[['strike', 'mark']].describe())
  print("\nRésumé statistique des maturités :")
  print(option_data['Maturity'].describe())

  print("\n✅ Voici les options utilisées pour la calibration :\n")

  for i, row in option_data.iterrows():
    print(f"Option {i + 1}: Strike={row['strike']:.2f}, Maturity={row['Maturity']:.4f} ans, " 
          f"Type={row['type']}, Mark={row['mark']:.4f}, "
          f"Last={row['last']:.4f}, Volume={row['volume']}, OI={row['open_interest']}")

    print("\n=== Fin de la liste des options calibrées ===\n")

  # Bornes pour les paramètres
  bounds = [(0.001, 2),  # v0 (vol de départ)
            (-0.99, 0.99),  # rho (correl donc entre -1 et 1)
            (0.001, 2),  # theta
            (0.01, 20),  # k (retour a la moyenne)
            (0.01, 7)]  # eta (vol of vol)

  # Plusieurs points de départ pour éviter les minima locaux
  initial_params_list = [
    [0.04, -0.7, 0.04, 1.0, 0.5],
    [0.1, -0.5, 0.1, 2.0, 0.3],
    [0.05, -0.3, 0.05, 3.0, 0.4]
  ]

  # Définition de la fonction objective simplifiée
  def obj_simplified(params):
    v0, rho, theta, k, eta = params
    return objective_function([v0, rho, theta, k, eta], option_data, S0, r)

  # Tester les valeurs initiales
  print("Valeur de la fonction objectif (initiale) :", obj_simplified(initial_params_list[0]))
  test_params = [0.08, -0.5, 0.06, 1.2, 0.8]
  print("Valeur de la fonction objectif (modifiée) :", obj_simplified(test_params))

  # Optimisation avec plusieurs points de départ
  best_result = None
  best_error = float('inf')

  for init_params in initial_params_list:
    print(f"\nEssai d'optimisation avec point de départ: {init_params}")
    result = minimize(obj_simplified, init_params, bounds=bounds, method='L-BFGS-B' )
    #result = differential_evolution(obj_simplified, bounds)
    print(f"Résultat intermédiaire: fun={result.fun}, success={result.success}")

    if result.fun < best_error:
      best_error = result.fun
      best_result = result
      print(f"Nouveau meilleur résultat trouvé! Erreur: {best_error}")

    if best_result is None or not best_result.success:
      print("Aucune optimisation n'a réussi. Retour des meilleurs paramètres trouvés.")
      if best_result is None:
        # Si aucun résultat, retourner les paramètres initiaux du premier essai
        return initial_params_list[0]

      # Retourner les meilleurs paramètres trouvés
    v0, rho, theta, k, eta = best_result.x
    return [v0, rho, theta, k, eta]



# Prix spot et taux sans risque
S0 = 215  # À ajuster selon votre actif
r = 0.045  # À ajuster selon le taux actuel

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