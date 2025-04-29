import pandas as pd
import numpy as np
from Call_Heston import heston_option_price_call,heston_option_price_put
from Modele_Heston import payoff_heston
from math import exp
from ZeroCoupon import ZeroCoupon
from Maturite import Maturite
import matplotlib.pyplot as plt


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

def option_barriere_call_DownandOut(S0, K,params_heston, barriere,T, r, Nmc=10000, N=252, seed=42):
    v0, rho, theta, k, eta = params_heston
    # Simulation de S(t) avec Heston
    S_paths = payoff_heston(r, T, K, S0=S0, rho=rho, theta=theta, k=k, eta=eta, N=N, Nmc=Nmc, seed=seed, v0=v0)

    crossed_barrier = np.min(S_paths, axis=1) <= barriere
    S_T = S_paths[:, -1]
    payoff_call = np.maximum(S_T - K, 0)

    # Payoff final : zéro si barrière non franchie
    payoffs = np.where(crossed_barrier,  0,payoff_call)

    discounted_payoffs = np.exp(-r * T) * payoffs
    prob_activation = np.mean(crossed_barrier)
    # Prix estimé
    price = np.mean(discounted_payoffs)
    return price,prob_activation

def option_barriere_Put_UPandIn(S0, K,params_heston, barriere,T, r, Nmc=10000, N=252, seed=42):
    v0, rho, theta, k, eta = params_heston
    # Simulation de S(t) avec Heston
    S_paths = payoff_heston(r, T, K, S0=S0, rho=rho, theta=theta, k=k, eta=eta, N=N, Nmc=Nmc, seed=seed, v0=v0)

    crossed_barrier = np.max(S_paths, axis=1) >= barriere
    S_T = S_paths[:, -1]
    payoff_put = np.maximum(K- S_T, 0)

    # Payoff final : zéro si barrière non franchie
    payoffs = np.where(crossed_barrier, payoff_put, 0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    prob_activation = np.mean(crossed_barrier)

    # Prix estimé
    price = np.mean(discounted_payoffs)
    return price,prob_activation

def option_barriere_put_UPandOut(S0, K,params_heston, barriere,T, r, Nmc=10000, N=252, seed=42):
    v0, rho, theta, k, eta = params_heston
    # Simulation de S(t) avec Heston
    S_paths = payoff_heston(r, T, K, S0=S0, rho=rho, theta=theta, k=k, eta=eta, N=N, Nmc=Nmc, seed=seed, v0=v0)

    crossed_barrier = np.max(S_paths, axis=1) >= barriere
    S_T = S_paths[:, -1]
    payoff_put = np.maximum(K- S_T, 0)

    # Payoff final : zéro si barrière non franchie
    payoffs = np.where(crossed_barrier,  0,payoff_put)

    discounted_payoffs = np.exp(-r * T) * payoffs
    prob_activation = np.mean(crossed_barrier)
    # Prix estimé
    price = np.mean(discounted_payoffs)
    return price,prob_activation



def price_reverse_convertible(S0, params_heston, coupon_rate, protection_barrier, r, maturite, nominal=1000, method="continu", Nmc=10000, N_steps=252):
    v0, rho, theta, k, eta = params_heston

    T = maturite.maturite_en_annees

    # Prix du Zero Coupon
    zc = ZeroCoupon("ZeroCoupon RC", r, maturite, nominal, method)
    price_zc = zc.prix()

    # Prix du put (perte si chute sous la barrière)
    put_price = heston_option_price_put(r, T, protection_barrier * S0, S0, rho, theta, k, eta, v0, Nmc, N_steps)

    # Coupon payé
    coupon_value = coupon_rate * nominal * exp(-r * T)

    # Prix total Reverse Convertible
    reverse_price = price_zc + coupon_value - (put_price * nominal / S0)

    return reverse_price


def Note_Capital_Proteger(S0, params_heston, rebate, barriere, K, r, maturite,
                          nominal=1000, method="continu", Nmc=10000, N_steps=252, seed=42):
    v0, rho, theta, k, eta = params_heston
    T = maturite.maturite_en_annees

    # Prix du Zero Coupon
    zc = ZeroCoupon("ZeroCoupon RC", r, maturite, nominal, method)
    price_zc = zc.prix()

    # Prix du call vanille et du call knock-out
    call_price = heston_option_price_call(r, T, K, S0, rho, theta, k, eta, v0, Nmc, N_steps)
    call_KO = option_barriere_call_UPandOut(S0, K, params_heston, barriere, T, r, Nmc=Nmc, N=N_steps, seed=seed)

    # Prix total de la note à capital protégé
    note_price = price_zc + call_price - call_KO + rebate

    return note_price

def plot_payoff_note(S0, K, barriere, rebate, params_heston, maturite, r, nominal=100):
    """
    Trace le payoff simulé de la Note à Capital Protégé.
    """
    S_T_range = np.linspace(0.5 * S0, 1.5 * S0, 50)  # seulement 50 points car simulation lourde
    payoffs = []

    for S_T in S_T_range:
        payoff = Note_Capital_Proteger(S_T, params_heston, rebate, barriere, K, r, maturite, nominal, method="continu", Nmc=3000, N_steps=100, seed=42)
        payoffs.append(payoff)

    plt.figure(figsize=(10,6))
    plt.plot(S_T_range, payoffs, lw=2, label="Prix de la Note")
    plt.axhline(nominal + rebate, color='gray', linestyle='--', label="Capital protégé + rebate")
    plt.axvline(barriere, color='red', linestyle='--', label="Barrière KO")
    plt.axvline(K, color='blue', linestyle='--', label="Strike")
    plt.title("Payoff d'une Note à Capital Protégé (avec simulation Heston)")
    plt.xlabel("Prix final de l'action $S_T$")
    plt.ylabel("Prix (€)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from datetime import datetime

    # Paramètres
    S0 = 215
    params_heston = [0.117478, -0.353219, 0.063540, 1.251697, 0.898913]
    coupon_rate = 0.10
    protection_barrier = 0.70
    r = 0.045
    nominal = 1

    # Maturité
    date_aujourdhui = datetime(2025, 3, 20).strftime("%Y-%m-%d")
    date_maturite = datetime(2026, 3, 20).strftime("%Y-%m-%d")
    maturite = Maturite(date_aujourdhui, date_maturite, "Act/365")

    prix_RC = price_reverse_convertible(S0, params_heston, coupon_rate, protection_barrier, r, maturite, nominal)
    print(f"Prix de la Reverse Convertible : {prix_RC:.2f} EUR")


if __name__ == "__main__":
    from datetime import datetime
    from Maturite import Maturite

    # Paramètres
    S0 = 215
    params_heston = [0.117478, -0.353219, 0.063540, 1.251697, 0.898913]
    coupon_rate = 0.10
    protection_barrier = 220
    r = 0.045
    nominal = 100

    # Maturité
    date_aujourdhui = datetime(2025, 3, 20).strftime("%Y-%m-%d")
    date_maturite = datetime(2026, 3, 20).strftime("%Y-%m-%d")
    maturite = Maturite(date_aujourdhui, date_maturite, "Act/365")

    plot_payoff_note(S0=215, K=215, barriere=220, rebate=5, params_heston=params_heston, maturite=maturite, r=r, nominal=1000)
