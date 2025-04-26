from Modele Heston import payoff_heston
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


### L'objectif ici  étant de pricer un call avec le modele de Heston

# On commence par recuperer les parametre calibré par notre modele puis on va simuler S et V en meme temps.

# Une fois cela effectué on pourra par MC calculter S_T donc S[-1] et farie (S_T-K) et faire la moyenne dess simulation,
# il faudra faire des test pour que ca dure un peu moins de 1 minutes sinon ca sera trop long (call spread, strip strap ect prendront
# trop de temps)



def heston_option_PUT(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42):
  # Fixer le seed pour la reproductibilité
  np.random.seed(seed)
  somme = 0
  total_payoff = 0

    for _ in range(n_simulations):
        # Générer une trajectoire
        S = payoff_heston(r, T, K, S0, rho, theta, k, eta, N, v0)[0]

        # Prix final de l'actif
        S_T = S[-1]

        # Calculer le payoff d'un put

        payoff = max(K - S_T, 0)
        somme+=payoff
    resultat = exp(-r*T)*somme/(n_simulations)
    return resultat

def heston_option_Call(r, T, K, S0, rho, theta, k, eta, v0, n_simulations=1000, N=100, option_type='call', seed=42):
    # Fixer le seed pour la reproductibilité
    np.random.seed(seed)

    somme = 0
    for _ in range(n_simulations):
        # Générer une trajectoire
        S = payoff_heston(r, T, K, S0, rho, theta, k, eta, N, v0)[0]

        # Prix final de l'actif
        S_T = S[-1]

        # Calculer le payoff

        payoff = max(S_T - K, 0)
        somme += payoff
    resultat = exp(-r * T) * somme / (n_simulations)
    return resultat


