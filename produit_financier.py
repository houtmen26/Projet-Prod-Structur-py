import pandas as pd
import numpy as np
from abc import ABC,abstractmethod
from Call_Heston import heston_option_price_call,heston_option_price_put
from dataclasses import dataclass

@dataclass
class MonteCarloConfig: #
    Nmc: int = 1000
    N: int = 100
    seed: int = None

class Action:
    def __init__(self, nom: str, S0: float):
        self.nom = nom
        self.S0 = S0


class ProduitFinancier(ABC):
    def __init__(self, sous_jacent: Action, maturite: float,parametres: float,r:float):
        self.sous_jacent = sous_jacent
        self.maturite = maturite
        self.parametres = parametres # Parametre du modele de heston (tableau)
        self.r = r
    @abstractmethod
    def price(self, modele):
        """
        Méthode abstraite pour calculer le prix du produit financier. Cette méthode doit
        être implémentée dans les classes dérivées pour chaque type de produit.
        """
        pass

class HestonOptionMixin:
    def __init__(self, mc_config: MonteCarloConfig = None):
        self.mc_config = mc_config or MonteCarloConfig()

class Call(ProduitFinancier, HestonOptionMixin):
    def __init__(self, sous_jacent: Action, maturite: float, parametres: list,
                 r: float, strike: float, mc_config=None):
        ProduitFinancier.__init__(self, sous_jacent, maturite, parametres, r)
        HestonOptionMixin.__init__(self, mc_config)
        self.strike = strike

    def price(self):
        v0, rho, theta, k, eta = self.parametres
        return heston_option_price_call(
            r=self.r,
            T=self.maturite,
            K=self.strike,
            S0=self.sous_jacent.S0,
            rho=rho,
            theta=theta,
            k=k,
            eta=eta,
            v0=v0,
            Nmc=self.mc_config.Nmc,
            N=self.mc_config.N,
            seed=self.mc_config.seed
        )

class Put(ProduitFinancier, HestonOptionMixin):
    def __init__(self, sous_jacent: Action, maturite: float, parametres: list,
                 r: float, strike: float, mc_config=None):
        ProduitFinancier.__init__(self, sous_jacent, maturite, parametres, r)
        HestonOptionMixin.__init__(self, mc_config)
        self.strike = strike

    def price(self):
        v0, rho,theta, k, eta = self.parametres
        return heston_option_price_put(
            r=self.r,
            T=self.maturite,
            K=self.strike,
            S0=self.sous_jacent.S0,
            rho=rho,
            theta=theta,
            k=k,
            eta=eta,
            v0=v0,
            Nmc=self.mc_config.Nmc,
            N=self.mc_config.N,
            seed=self.mc_config.seed
        )
# Paramètres complets pour le modèle de Heston

# Configuration des paramètres
param = [0.117478, -0.353219, 0.063540, 1.251697, 0.898913 ]
taux_sans_risque = 0.045

# Configuration Monte Carlo personnalisée
mc_config = MonteCarloConfig(Nmc=5000, N=200, seed=42)

# Création des options
action = Action("AAPL", 215)
call = Call(action, 1.0, param, taux_sans_risque, 220, mc_config)
put = Put(action, 1.0, param, taux_sans_risque, 220, mc_config)

# Calcul des prix
print(f"Call price: {call.price()}")
print(f"Put price: {put.price()}")