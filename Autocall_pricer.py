import pandas as pd
import numpy as np
from Call_Heston import heston_option_price_call,heston_option_price_put
from Modele_Heston import payoff_heston
from math import exp

def price_autocall_heston(S0, params_heston, autocall_params, r, Nmc=10000, N_steps=252, seed=42):
    v0, rho, theta, k, eta = params_heston

    observation_dates = autocall_params['observation_dates']  # En années (ex: [0.25, 0.5, 0.75, 1])
    barriers = autocall_params['barriers']  # Ex: [1.0, 1.0, 1.0, 1.0]
    coupons = autocall_params['coupons']  # Ex: [0.02, 0.04, 0.06, 0.08]
    protection_barrier = autocall_params['protection_barrier']  # Ex: 0.6

    # Vérification de cohérence
    assert len(observation_dates) == len(barriers) == len(coupons), "Incohérence entre dates/barrières/coupons"

    # Construction de la grille temporelle
    T_max = max(observation_dates)
    dt = T_max / N_steps
    time_grid = np.linspace(0, T_max, N_steps + 1)
    observation_indices = [np.argmin(np.abs(time_grid - t)) for t in observation_dates]

    # Simulation de S(t) avec Heston
    S_paths = payoff_heston(r, T_max, K=0, S0=S0, rho=rho, theta=theta, k=k, eta=eta, N=N_steps, Nmc=Nmc, seed=seed, v0=v0)

    # Initialisation
    payoffs = np.zeros(Nmc)
    exercised = np.zeros(Nmc, dtype=bool)

    # Observation des barrières
    for idx_obs, (t_idx, barrier, coupon) in enumerate(zip(observation_indices, barriers, coupons)):
        mask = (S_paths[:, t_idx] >= barrier * S0) & (~exercised)
        payoffs[mask] = (1 + coupon) * exp(-r * observation_dates[idx_obs])
        exercised[mask] = True

    # À maturité
    mask_final = ~exercised
    final_prices = S_paths[:, -1]

    payoffs[mask_final & (final_prices >= protection_barrier * S0)] = exp(-r * T_max)
    payoffs[mask_final & (final_prices < protection_barrier * S0)] = (final_prices[mask_final & (final_prices < protection_barrier * S0)] / S0) * exp(-r * T_max)

    # Prix de l'autocall
    return np.mean(payoffs)


if __name__ == "__main__":
    random_seed = np.random.randint(0, 1000000)
    autocall_params = {
        'observation_dates': [0.25, 0.5, 0.75, 1],  # tous les 3 mois
        'barriers': [1.0, 1.0, 1.0, 1.0],            # barrière 100% S0
        'coupons': [0.02, 0.04, 0.06, 0.08],          # coupons cumulés
        'protection_barrier': 0.6                    # protection à 60%
    }
    S0 = 215
    r = 0.045
    param = [0.117478, -0.353219, 0.063540, 1.251697, 0.898913 ]


    prix_autocall = price_autocall_heston(S0, param, autocall_params, r, Nmc=100000, N_steps=252, seed=random_seed )
    print(f"Le prix de l'autocall est : {prix_autocall:.4f}")




def option_barriere_call_UPandIn(S0, K,params_heston, barriere,T, r, Nmc=10000, N=252, seed=42):
    v0, rho, theta, k, eta = params_heston
    # Simulation de S(t) avec Heston
    S_paths = payoff_heston(r, T, K, S0=S0, rho=rho, theta=theta, k=k, eta=eta, N=N, Nmc=Nmc, seed=seed, v0=v0)

    crossed_barrier = np.max(S_paths, axis=1) >= barriere
    S_T = S_paths[:, -1]
    payoff_call = np.maximum(S_T - K, 0)

    # Payoff final : zéro si barrière non franchie
    payoffs = np.where(crossed_barrier, payoff_call, 0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    prob_activation = np.mean(crossed_barrier)

    # Prix estimé
    price = np.mean(discounted_payoffs)
    return price,prob_activation

def option_barriere_call_UPandOut(S0, K,params_heston, barriere,T, r, Nmc=10000, N=252, seed=42):
    v0, rho, theta, k, eta = params_heston
    # Simulation de S(t) avec Heston
    S_paths = payoff_heston(r, T, K, S0=S0, rho=rho, theta=theta, k=k, eta=eta, N=N, Nmc=Nmc, seed=seed, v0=v0)

    crossed_barrier = np.max(S_paths, axis=1) >= barriere
    S_T = S_paths[:, -1]
    payoff_call = np.maximum(S_T - K, 0)

    # Payoff final : zéro si barrière non franchie
    payoffs = np.where(crossed_barrier,  0,payoff_call)

    discounted_payoffs = np.exp(-r * T) * payoffs
    prob_activation = np.mean(crossed_barrier)
    # Prix estimé
    price = np.mean(discounted_payoffs)
    return price,prob_activation
