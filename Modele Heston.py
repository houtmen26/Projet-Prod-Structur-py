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
    v[i] = max(0, v[i-1] + k*(theta - v[i-1])*delta_t + neta*sqrt_v*sqrt(delta_t)*X + 0.25*neta**2*delta_t*(X**2 - 1))
    vol[i] = sqrt(abs(v[i]))
  return S, v, vol

def heston_option_price(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42):
  # Fixer le seed pour la reproductibilité
  np.random.seed(seed)

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
  constant_seed = 12345

  for i, row in enumerate(option_data.itertuples(), start=1):
    K = row.strike
    T = row.Maturity
    market_price = row.mark
    option_type = row.type

    if pd.isna(market_price):
      continue

    print(f"Calibration sur option {i}/{len(option_data)}")

    # Prix selon le modèle - AUGMENTER le nombre de simulations et UTILISER le seed
    model_price = heston_option_price(r, T, K, S0, rho, theta, k, eta, v0,
                                      n_simulations=5000, N=100, option_type=option_type, seed=constant_seed)

    # Pour le debug, afficher les prix
    print(f"  Strike={K}, T={T}, Market={market_price:.4f}, Model={model_price:.4f}")

    # Éviter les divisions par zéro ou par des valeurs très petites
    denominator = max(0.01, abs(market_price))
    error = ((model_price - market_price) / denominator) ** 2

    weight = 1.0
    if abs(K / S0 - 1) < 0.05:  # ATM
      weight *= 2.0
    if T < 0.25:  # Court terme
      weight *= 1.5

    total_error += error * weight

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
    result = minimize(obj_simplified, init_params, bounds=bounds,
                      method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})

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
S0 = 215.0  # À ajuster selon votre actif
r = 0.05  # À ajuster selon le taux actuel

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