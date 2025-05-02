
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
from produit_financier import *
from Main import maturite
from Modele_Heston import payoff_heston
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.stats import norm
import streamlit as st

import streamlit as st
import matplotlib.pyplot as plt
import sys
def st_show(use_st=True):
    """Affiche dans Streamlit ou en fenêtre selon besoin"""
    if use_st and 'streamlit' in sys.modules:
        st.pyplot(plt.gcf())
    else:
        try:
            plt.show()
        except:
            plt.savefig('temp_plot.png')
            st.image('temp_plot.png')
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




def bs_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def implied_volatility(price, S, K, T, r, option_type="call"):
    try:
        return brentq(lambda sigma: bs_price(S, K, T, r, sigma, option_type) - price, 1e-5, 3.0)
    except:
        return np.nan  # Retourne NaN si pas de solution trouvée



def trace_smile_vol_heston(S0, r, maturite, option_type, params, strikes, Nmc=20000):
  v0, rho, theta, k, eta = params
  vols_implicites = []
  prix_heston = []

  for K in strikes:
    price = heston_option_price_call(
      r=r, T=maturite, K=K, S0=S0, rho=rho, theta=theta,
      k=k, eta=eta, v0=v0, Nmc=Nmc, N=100,  seed=42
    )
    iv = implied_volatility(price, S0, K, maturite, r, option_type)
    prix_heston.append(price)
    vols_implicites.append(iv)

  # Affichage
  plt.figure(figsize=(10, 6))
  plt.plot(strikes, vols_implicites, 'o-', color='darkblue')
  plt.axvline(S0, linestyle='--', color='gray', label='Spot')
  plt.xlabel("Strike")
  plt.ylabel("Volatilité implicite (%)")
  plt.title(f"Smile de volatilité ({option_type.upper()}, maturité = {maturite} an)")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()

  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(strikes, vols_implicites, 'o-', color='darkblue')
  ax.axvline(S0, linestyle='--', color='gray', label='Spot')
  ax.set_xlabel("Strike")
  ax.set_ylabel("Volatilité implicite (%)")
  ax.set_title(f"Smile de volatilité ({option_type.upper()}, maturité = {maturite} an)")
  ax.grid(True)
  ax.legend()
  st_show()


  return pd.DataFrame({'Strike': strikes, 'Prix Heston': prix_heston, 'Vol Implicite': vols_implicites})


if __name__ == "__main__":
  # Prix spot et taux sans risque
  S0 = 215  # À ajuster selon votre actif
  r = 0.045  # À ajuster selon le taux actuel
  Nmc=10000
  N=100
  maturite=1
  strike = np.linspace(S0 * 0.7, S0 * 1.3, 20)

  params=[0.117478, -0.353219, 0.063540, 1.251697, 0.898913 ]

  #trace_smile_vol_heston(S0, r, maturite, option_type="call",  params=params, strikes=strike, Nmc=20000)
  # Création de l'instance de Call
  mc_config = MonteCarloConfig(Nmc=Nmc, N=N, seed=42)
  action = Action("Apple", S0)  # Nom de l'action et prix initial
  strike=230



  # Configuration
  S0 = 100
  r = 0.03
  maturite = 1
  params = [0.04, -0.7, 0.04, 1.0, 0.2]
  K = [90, 110]
  Nmc = 10000
  N = 100

  # Création de l'instance
  mc_config = MonteCarloConfig(Nmc=Nmc, N=N, seed=42)
  action = Action("Asset", S0)
  call_spread = CallSpread(action, maturite, params, r, K, mc_config)

  # Calcul du prix
  prix = call_spread.plot_payoff()

  strike = 100
  straddle = Straddle(action, maturite, params, r, [strike], mc_config)
  prix = straddle.plot_payoff()


  K1, K2 = 90, 110
  strangle = Strangle(action, maturite, params, r, [K1, K2], mc_config)
  prix = strangle.plot_payoff()


  strike = 100
  strip = Strip(action, maturite, params, r, [strike], mc_config)
  prix = strip.plot_payoff()


  strike = 100
  strap = Strap(action, maturite, params, r, [strike], mc_config)
  prix = strap.plot_payoff()

  strike = 100
  h1 = 0.5  # 0.5% du strike
  binary_call = OptionCallBinaire(action, maturite, params, r, strike, h1, mc_config)
  prix = binary_call.plot_payoff()


  strike = 100
  barriere = 120
  barrier_call = OptionBarriereUpAndIn_CALL(action, maturite, params, r, strike, barriere, mc_config)
  prix = barrier_call.plot_payoff()



  strike = 100
  barriere = 120
  nominal = 1000
  rebate = 20
  note = NoteCapitalProtegee(action, maturite, params, r, strike, barriere, rebate=rebate,nominal=nominal, mc_config=mc_config)
  prix = note.plot_payoff()


  coupon_rate = 0.1  # 10%
  protection_barrier = 0.7  # 70% du spot
  rc = ReverseConvertible(action, maturite, params, r, coupon_rate, protection_barrier, nominal=1000,
                          mc_config=mc_config)
  prix = rc.plot_payoff()










