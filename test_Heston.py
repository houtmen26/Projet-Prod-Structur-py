import numpy as np
import pandas as pd
import numpy.random as nrd
import matplotlib.pyplot as plt
from math import *
import scipy.stats as si
from scipy.optimize import minimize
from numba import jit
import time


# Objectif est de définir le modele d'Heston et de le calibrer

# Optimisation du code avec Numba pour accélération
@jit(nopython=True)
def payoff_heston_optimized(r, T, K, S0, rho, theta, k, neta, N, v0):
    delta_t = T / N
    S = np.zeros(N)
    v = np.zeros(N)
    vol = np.zeros(N)
    S[0] = S0
    v[0] = v0
    vol[0] = sqrt(v0)

    for i in range(1, N):
        X = nrd.randn()
        X1 = nrd.randn()
        S[i] = S[i - 1] * exp((r - 0.5 * v[i - 1]) * delta_t + sqrt(v[i - 1]) * (
                    rho * sqrt(delta_t) * X + sqrt(1 - rho ** 2) * sqrt(delta_t) * X1))
        v[i] = max(1e-8, v[i - 1] + k * (theta - v[i - 1]) * delta_t + neta * sqrt(max(1e-8, v[i - 1])) * sqrt(
            delta_t) * X + neta ** 2 / 4 * delta_t * (X ** 2 - 1))
        vol[i] = sqrt(v[i])

    return S[-1]  # Retourne uniquement le prix final


@jit(nopython=True)
def heston_option_price_optimized(r, T, K, S0, rho, theta, k, eta, v0, n_simulations, N, option_type_call):
    total_payoff = 0.0

    for _ in range(n_simulations):
        # Génération d'une trajectoire et récupération du prix final
        S_T = payoff_heston_optimized(r, T, K, S0, rho, theta, k, eta, N, v0)

        # Calcul du payoff
        if option_type_call:
            payoff = max(S_T - K, 0)
        else:
            payoff = max(K - S_T, 0)

        total_payoff += payoff

    # Prix moyen actualisé
    option_price = exp(-r * T) * (total_payoff / n_simulations)
    return option_price


def process_data(filename):
    """Charge et prépare les données d'options"""
    option_data = pd.read_csv(filename, sep=';')
    print('Données chargées :', option_data)

    # Convertir les dates en maturités
    option_data['Maturity'] = (pd.to_datetime(option_data['expiration']) -
                               pd.to_datetime(option_data['price_date'])).dt.days / 365.0

    print("Maturités calculées :", option_data['Maturity'])
    print("Volatilités implicites :", option_data['implied_volatility'])

    # Conversion des types d'options en booléens pour numba
    option_data['is_call'] = option_data['type'].str.lower() == 'call'

    return option_data


def objective_function(params, option_data, S0, r, n_sim=1000, N=50):
    """Fonction objectif optimisée pour la calibration"""
    v0, rho, theta, k, eta = params

    # Vérifier que les paramètres sont dans les limites physiques
    if v0 <= 0 or theta <= 0 or k <= 0 or eta <= 0 or abs(rho) >= 1:
        return 1e10

    total_error = 0.0

    # Utiliser un sous-ensemble pour accélérer la calibration si nécessaire
    if len(option_data) > 10:
        sample_data = option_data.sample(n=min(10, len(option_data)), random_state=42)
    else:
        sample_data = option_data

    for _, row in sample_data.iterrows():
        K = row['strike']
        T = row['Maturity']
        market_price = row['mark']
        is_call = row['is_call']

        # Calculer le prix avec le modèle
        model_price = heston_option_price_optimized(r, T, K, S0, rho, theta, k, eta, v0,
                                                    n_sim, N, is_call)

        # Calcul de l'erreur (pondérée par le prix de marché pour donner plus d'importance aux options ATM)
        weight = 1.0 / max(0.1, abs(market_price))
        error = ((model_price - market_price) * weight) ** 2
        total_error += error

    return total_error


def calibrate_heston_simplified(option_data, S0, r):
    """Calibration à 3 paramètres (v0, rho, theta) avec k et eta fixés"""
    start_time = time.time()

    # Paramètres fixes
    k_fixed = 2.0
    eta_fixed = 0.3

    # Estimation de v0 à partir de la volatilité implicite moyenne des options ATM
    ATM_options = option_data[abs(option_data['strike'] - S0) / S0 < 0.05]
    if not ATM_options.empty:
        v0_initial = (ATM_options['implied_volatility'].mean()) ** 2
    else:
        v0_initial = 0.04

    # Valeurs initiales pour v0, rho, theta
    initial_params = [v0_initial, -0.7, v0_initial]

    # Bornes pour ces paramètres
    bounds = [(0.001, 0.25),  # v0
              (-0.95, 0.5),  # rho (typiquement négatif pour les actions)
              (0.001, 0.25)]  # theta

    def obj_simplified(params):
        v0, rho, theta = params
        return objective_function([v0, rho, theta, k_fixed, eta_fixed], option_data, S0, r)

    # Optimisation avec nombre réduit d'itérations pour la rapidité
    result = minimize(obj_simplified, initial_params, bounds=bounds,
                      method='L-BFGS-B', options={'maxiter': 20})

    v0, rho, theta = result.x

    end_time = time.time()
    print(f"Calibration terminée en {end_time - start_time:.2f} secondes")

    return [v0, rho, theta, k_fixed, eta_fixed]


def validate_calibration(params, option_data, S0, r):
    """Valide les résultats de la calibration en comparant prix modèle vs marché"""
    v0, rho, theta, k, eta = params

    results = []
    for _, row in option_data.iterrows():
        K = row['strike']
        T = row['Maturity']
        market_price = row['mark']
        is_call = row['is_call']

        model_price = heston_option_price_optimized(r, T, K, S0, rho, theta, k, eta, v0,
                                                    5000, 100, is_call)

        results.append({
            'Strike': K,
            'Maturity': T,
            'Type': 'call' if is_call else 'put',
            'Market': market_price,
            'Model': model_price,
            'Error': model_price - market_price,
            'Error%': 100 * (model_price - market_price) / market_price
        })

    return pd.DataFrame(results)


def main():
    start_time = time.time()

    # Chargement des données
    option_data = process_data("options.csv")

    # Prix spot (à ajuster selon vos données)
    S0 = float(option_data['underlying_price'].iloc[0])
    print(f"Prix spot S0: {S0}")

    # Taux sans risque (à ajuster)
    r = 0.01

    # Warmup de JIT (première exécution compile les fonctions)
    print("Initialisation de Numba JIT...")
    _ = heston_option_price_optimized(r, 1.0, 100.0, S0, -0.7, 0.04, 2.0, 0.3, 0.04, 100, 50, True)

    # Calibration
    print("Lancement de la calibration...")
    optimal_params = calibrate_heston_simplified(option_data, S0, r)
    v0, rho, theta, k, eta = optimal_params

    print(f"\nParamètres calibrés:")
    print(f"v0 = {v0:.6f} (variance initiale)")
    print(f"rho = {rho:.6f} (corrélation)")
    print(f"theta = {theta:.6f} (variance long-terme)")
    print(f"k = {k:.6f} (vitesse de retour)")
    print(f"eta = {eta:.6f} (vol de vol)")

    # Validation des résultats
    results = validate_calibration(optimal_params, option_data, S0, r)
    print("\nComparaison des prix (échantillon):")
    print(results[['Strike', 'Maturity', 'Type', 'Market', 'Model', 'Error%']].head())

    # Affichage de l'erreur moyenne absolue en pourcentage
    mae = results['Error'].abs().mean()
    mape = results['Error%'].abs().mean()
    print(f"\nErreur moyenne absolue: {mae:.4f}")
    print(f"Erreur moyenne absolue en pourcentage: {mape:.2f}%")

    # Graphique des résultats
    plt.figure(figsize=(12, 10))

    # Prix par Strike
    plt.subplot(2, 1, 1)
    for mat in sorted(results['Maturity'].unique()):
        subset = results[results['Maturity'] == mat]
        plt.scatter(subset['Strike'], subset['Market'], marker='o', label=f'Marché (T={mat:.2f})')
        plt.scatter(subset['Strike'], subset['Model'], marker='x', label=f'Modèle (T={mat:.2f})')

    plt.axvline(x=S0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Strike')
    plt.ylabel('Prix')
    plt.title('Comparaison prix marché vs modèle calibré')
    plt.legend()

    # Erreur relative par Strike
    plt.subplot(2, 1, 2)
    for mat in sorted(results['Maturity'].unique()):
        subset = results[results['Maturity'] == mat]
        plt.scatter(subset['Strike'], subset['Error%'], marker='o', label=f'T={mat:.2f}')

    plt.axhline(y=0, color='r', linestyle='-')
    plt.axvline(x=S0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Strike')
    plt.ylabel('Erreur relative (%)')
    plt.title('Erreur relative du modèle calibré')
    plt.legend()

    plt.tight_layout()
    plt.savefig('heston_calibration_results.png')
    plt.show()

    end_time = time.time()
    print(f"\nTemps total d'exécution: {end_time - start_time:.2f} secondes")


if __name__ == "__main__":
    main()