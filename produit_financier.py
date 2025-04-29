import pandas as pd
import numpy as np
from abc import ABC,abstractmethod
from Call_Heston import heston_option_price_call,heston_option_price_put
from dataclasses import dataclass
from Modele_Heston import payoff_heston
from math import *
import matplotlib.pyplot as plt

from ZeroCoupon import ZeroCoupon

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
    def price(self):
        """
        Méthode abstraite pour calculer le prix du produit financier. Cette méthode doit
        être implémentée dans les classes dérivées pour chaque type de produit.
        """
        pass

    #@abstractmethod
    #def plot_payoff(self):
        """
        Méthode à implémenter dans les classes enfants pour afficher le payoff.
        """
     #   pass

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

    def plot_payoff(self):
        S_range = np.linspace(0.5 * self.strike, 1.5 * self.strike, 100)
        payoff = np.maximum(S_range - self.strike, 0)
        plt.plot(S_range, payoff, label="Call Payoff")
        plt.axvline(self.strike, color='gray', linestyle='--', label='Strike')
        plt.title("Payoff d'un Call")
        plt.xlabel("Prix du sous-jacent")
        plt.ylabel("Payoff à maturité")
        plt.grid(True)
        plt.legend()
        plt.show()

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

class StrategieOption(ProduitFinancier, HestonOptionMixin):
    """Classe de base pour les stratégies optionnelles composites (spread, straddle, etc.)."""
    def __init__(self, sous_jacent: Action, maturite: float, parametres: list,
                 r: float, strikes: list, mc_config=None):
        ProduitFinancier.__init__(self, sous_jacent, maturite, parametres, r)
        HestonOptionMixin.__init__(self, mc_config)
        self.strikes = strikes  # Liste de strikes

    @abstractmethod
    def price(self):
        pass


class CallSpread(StrategieOption):
    def price(self):
        v0, rho, theta, k, eta = self.parametres
        K1, K2 = self.strikes
        call1 = heston_option_price_call(self.r, self.maturite, K1, self.sous_jacent.S0,
                                         rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        call2 = heston_option_price_call(self.r, self.maturite, K2, self.sous_jacent.S0,
                                         rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        return call1 - call2


class Straddle(StrategieOption):
    def price(self):
        v0, rho, theta, k, eta = self.parametres
        K = self.strikes[0]
        call = heston_option_price_call(self.r, self.maturite, K, self.sous_jacent.S0,
                                        rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        put = heston_option_price_put(self.r, self.maturite, K, self.sous_jacent.S0,
                                      rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        return call + put

class Strangle(StrategieOption):
    def price(self):
        v0, rho, theta, k, eta = self.parametres
        K1, K2 = self.strikes
        call = heston_option_price_call(self.r, self.maturite, K1, self.sous_jacent.S0,
                                        rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        put = heston_option_price_put(self.r, self.maturite, K2, self.sous_jacent.S0,
                                      rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        return call + put

class Strip(StrategieOption):
    def price(self):
        v0, rho, theta, k, eta = self.parametres
        K = self.strikes[0]
        call = heston_option_price_call(self.r, self.maturite, K, self.sous_jacent.S0,
                                        rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        put = heston_option_price_put(self.r, self.maturite, K, self.sous_jacent.S0,
                                      rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        return call + 2 * put

class Strap(StrategieOption):
    def price(self):
        v0, rho, theta, k, eta = self.parametres
        K = self.strikes[0]
        call = heston_option_price_call(self.r, self.maturite, K, self.sous_jacent.S0,
                                        rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        put = heston_option_price_put(self.r, self.maturite, K, self.sous_jacent.S0,
                                      rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        return 2 * call + put


class OptionCallBinaire(StrategieOption):
    def __init__(self, sous_jacent, maturite, parametres, r, strike, h1, mc_config=None):
        super().__init__(sous_jacent, maturite, parametres, r, [strike], mc_config)
        self.h1 = h1

    def price(self):
        v0, rho, theta, k, eta = self.parametres
        K = self.strikes[0]
         # Dans la littérature il est dit qu'une bonne valeur de H1 est 0,5% du strike
        self.h1 = (0.5/100)*self.sous_jacent.S0
        call1 = heston_option_price_call(self.r, self.maturite, K, self.sous_jacent.S0,
                                         rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        call2 = heston_option_price_call(self.r, self.maturite, K + self.h1, self.sous_jacent.S0,
                                         rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        prix = (call1 - call2)/self.h1
        return prix



class Autocall(ProduitFinancier, HestonOptionMixin):
    def __init__(self, sous_jacent: Action, maturite:float, parametres: list, r: float,
                 autocall_params: dict, mc_config: MonteCarloConfig = None):
        ProduitFinancier.__init__(self, sous_jacent, maturite, parametres, r)
        HestonOptionMixin.__init__(self, mc_config)
        self.autocall_params = autocall_params

    def price(self):
        v0, rho, theta, k, eta = self.parametres

        observation_dates = self.autocall_params['observation_dates']
        barriers = self.autocall_params['barriers']
        coupons = self.autocall_params['coupons']
        protection_barrier = self.autocall_params['protection_barrier']

        assert len(observation_dates) == len(barriers) == len(coupons), "Incohérence entre dates/barrières/coupons"

        T_max = max(observation_dates)
        time_grid = np.linspace(0, T_max, self.mc_config.N + 1)
        observation_indices = [np.argmin(np.abs(time_grid - t)) for t in observation_dates]

        S_paths = payoff_heston(self.r, T_max, K=0, S0=self.sous_jacent.S0,
                                rho=rho, theta=theta, k=k, eta=eta,
                                N=self.mc_config.N, Nmc=self.mc_config.Nmc, seed=self.mc_config.seed, v0=v0)

        payoffs = np.zeros(self.mc_config.Nmc)
        exercised = np.zeros(self.mc_config.Nmc, dtype=bool)

        for idx_obs, (t_idx, barrier, coupon) in enumerate(zip(observation_indices, barriers, coupons)):
            mask = (S_paths[:, t_idx] >= barrier * self.sous_jacent.S0) & (~exercised)
            payoffs[mask] = (1 + coupon) * exp(-self.r * observation_dates[idx_obs])
            exercised[mask] = True

        mask_final = ~exercised
        final_prices = S_paths[:, -1]

        payoffs[mask_final & (final_prices >= protection_barrier * self.sous_jacent.S0)] = exp(-self.r * T_max)
        payoffs[mask_final & (final_prices < protection_barrier * self.sous_jacent.S0)] = (
            final_prices[mask_final & (final_prices < protection_barrier * self.sous_jacent.S0)] / self.sous_jacent.S0
        ) * exp(-self.r * T_max)

        return np.mean(payoffs)


class OptionBarriereUpAndIn_CALL(ProduitFinancier,HestonOptionMixin):
    def __init__(self, sous_jacent: Action, maturite:float, parametres:list, r:float,
                 K, barriere, mc_config: MonteCarloConfig = None):
        super().__init__(sous_jacent, maturite, parametres, r)
        HestonOptionMixin.__init__(self, mc_config)
        self.strike = K
        self.barriere = barriere


    def price(self):
        v0, rho, theta, k, eta = self.parametres
        S_paths = payoff_heston(self.r, self.maturite, K=0, S0=self.sous_jacent.S0,
                                rho=rho, theta=theta, k=k, eta=eta,
                                N=self.mc_config.N, Nmc=self.mc_config.Nmc, seed=self.mc_config.seed, v0=v0)

        crossed_barrier = np.max(S_paths, axis=1) >= self.barriere
        S_T = S_paths[:, -1]

        payoff_call = np.maximum(S_T - self.strike, 0)

        payoffs = np.where(crossed_barrier, payoff_call, 0)

        discounted_payoffs = np.exp(-self.r * self.maturite) * payoffs
        prob_activation = np.mean(crossed_barrier)

        # Prix estimé
        price = np.mean(discounted_payoffs)
        return price, prob_activation


class OptionBarriereUpAndOutCall(ProduitFinancier, HestonOptionMixin):
    """Option barrière Up-and-Out Call utilisant le modèle de Heston.

    L'option est désactivée si le sous-jacent dépasse la barrière.

    Attributes:
        strike (float): Prix d'exercice
        barriere (float): Niveau de désactivation (H > S0)
    """

    def __init__(self,
                 sous_jacent: Action,
                 maturite: float,
                 parametres: list,
                 r: float,
                 strike: float,
                 barriere: float,
                 mc_config: MonteCarloConfig = None):
        """
        Args:
            strike: Prix d'exercice (K)
            barriere: Niveau de désactivation (H > S0)
        """
        super().__init__(sous_jacent, maturite, parametres, r)
        HestonOptionMixin.__init__(self, mc_config)
        self.strike = strike
        self.barriere = barriere

    def price(self):
        """Calcule le prix et la probabilité de désactivation.

        Returns:
            tuple: (prix_estime, probabilite_desactivation)
        """
        v0, rho, theta, k, eta = self.parametres

        # Simulation des trajectoires
        S_paths = payoff_heston(
            self.r, self.maturite, K=0,  # K inutilisé ici
            S0=self.sous_jacent.S0,
            rho=rho, theta=theta, k=k, eta=eta,
            N=self.mc_config.N,
            Nmc=self.mc_config.Nmc,
            seed=self.mc_config.seed,
            v0=v0
        )

        max_paths = np.max(S_paths, axis=1)
        S_T = S_paths[:, -1]

        # Logique Up-and-Out : payoff seulement si la barrière N'EST PAS franchie
        payoff_call = np.maximum(S_T - self.strike, 0)
        payoffs = np.where(max_paths < self.barriere, payoff_call, 0)  # <-- Seule différence majeure

        # Actualisation
        discounted_payoffs = np.exp(-self.r * self.maturite) * payoffs
        prob_desactivation = np.mean(max_paths >= self.barriere)  # Probabilité que l'option soit désactivée

        return np.mean(discounted_payoffs), prob_desactivation


class OptionBarriereDownAndOutCall(ProduitFinancier, HestonOptionMixin):
    """Option barrière Down-and-Out Call. Désactivée si le sous-jacent tombe sous la barrière."""

    def __init__(self, sous_jacent: Action, maturite: float, parametres: list, r: float,
                 strike: float, barriere: float, mc_config: MonteCarloConfig = None):
        super().__init__(sous_jacent, maturite, parametres, r)
        HestonOptionMixin.__init__(self, mc_config)
        self.strike = strike
        self.barriere = barriere

    def price(self) :
        v0, rho, theta, k, eta = self.parametres
        S_paths = payoff_heston(self.r, self.maturite, K=0, S0=self.sous_jacent.S0,
                                rho=rho, theta=theta, k=k, eta=eta,
                                N=self.mc_config.N, Nmc=self.mc_config.Nmc,
                                seed=self.mc_config.seed, v0=v0)

        min_paths = np.min(S_paths, axis=1)
        S_T = S_paths[:, -1]

        payoff = np.maximum(S_T - self.strike, 0)
        payoffs = np.where(min_paths <= self.barriere, 0, payoff)  # Désactivé si barrière touchée

        discounted = np.exp(-self.r * self.maturite) * payoffs
        prob_desactivation = np.mean(min_paths <= self.barriere)

        return np.mean(discounted), prob_desactivation


class OptionBarriereUpAndInPut(ProduitFinancier, HestonOptionMixin):
    """Option barrière Up-and-In Put. Activée seulement si le sous-jacent dépasse la barrière."""

    def price(self) :
        v0, rho, theta, k, eta = self.parametres
        S_paths = payoff_heston(self.r, self.maturite, K=0, S0=self.sous_jacent.S0,
                                rho=rho, theta=theta, k=k, eta=eta,
                                N=self.mc_config.N, Nmc=self.mc_config.Nmc,
                                seed=self.mc_config.seed, v0=v0)

        max_paths = np.max(S_paths, axis=1)
        S_T = S_paths[:, -1]

        payoff = np.maximum(self.strike - S_T, 0)
        payoffs = np.where(max_paths >= self.barriere, payoff, 0)  # Activé si barrière franchie

        discounted = np.exp(-self.r * self.maturite) * payoffs
        prob_activation = np.mean(max_paths >= self.barriere)

        return np.mean(discounted), prob_activation


class OptionBarriereUpAndOutPut(ProduitFinancier, HestonOptionMixin):
    """Option barrière Up-and-Out Put. Désactivée si le sous-jacent dépasse la barrière."""

    def price(self) :
        v0, rho, theta, k, eta = self.parametres
        S_paths = payoff_heston(self.r, self.maturite, K=0, S0=self.sous_jacent.S0,
                                rho=rho, theta=theta, k=k, eta=eta,
                                N=self.mc_config.N, Nmc=self.mc_config.Nmc,
                                seed=self.mc_config.seed, v0=v0)

        max_paths = np.max(S_paths, axis=1)
        S_T = S_paths[:, -1]

        payoff = np.maximum(self.strike - S_T, 0)
        payoffs = np.where(max_paths >= self.barriere, 0, payoff)  # Désactivé si barrière franchie

        discounted = np.exp(-self.r * self.maturite) * payoffs
        prob_desactivation = np.mean(max_paths >= self.barriere)

        return np.mean(discounted), prob_desactivation



class NoteCapitalProtegee(ProduitFinancier, HestonOptionMixin):
    def __init__(self, sous_jacent: Action, maturite, parametres, r,
                 K, barriere, rebate=0, nominal=1000, method="continu",
                 mc_config: MonteCarloConfig = None):
        super().__init__(sous_jacent, maturite, parametres, r)
        HestonOptionMixin.__init__(self, mc_config)
        self.K = K
        self.barriere = barriere
        self.rebate = rebate
        self.nominal = nominal
        self.method = method

    def price(self):
        v0, rho, theta, k, eta = self.parametres
        T = self.maturite

        # Prix du ZC
        zc = ZeroCoupon("ZC Note Protégée", self.r, self.maturite, self.nominal, self.method)
        price_zc = zc.prix()

        # Prix du call standard

        call = Call(
            sous_jacent=self.sous_jacent,
            maturite=self.maturite,
            parametres=self.parametres,
            r=self.r,
            strike=self.strike,
            mc_config=self.mc_config
        ).price() * (self.nominal / self.sous_jacent.S0)

        # Call Knock-Out
        call_ko = OptionBarriereUpAndOutCall(
            sous_jacent=self.sous_jacent,
            maturite=self.maturite,
            parametres=self.parametres,
            r=self.r,
            strike=self.strike,
            barriere=self.barriere,
            mc_config=self.mc_config
        ).price()[0] * (self.nominal / self.sous_jacent.S0)

        rebate_discounted = self.rebate * np.exp(-self.r * T)


        return price_zc + call - call_ko + rebate_discounted


class ReverseConvertible(ProduitFinancier, HestonOptionMixin):
    def __init__(self, sous_jacent: Action, maturite:float, parametres: list, r: float,
                 coupon_rate: float, protection_barrier: float, nominal=1000,
                 method="continu", mc_config: MonteCarloConfig = None):
        ProduitFinancier.__init__(self, sous_jacent, maturite, parametres, r)
        HestonOptionMixin.__init__(self, mc_config)
        self.coupon_rate = coupon_rate
        self.protection_barrier = protection_barrier
        self.nominal = nominal
        self.method = method

    def price(self):
        v0, rho, theta, k, eta = self.parametres
        T = self.maturite

        zc = ZeroCoupon("ZeroCoupon RC", self.r, self.maturite, self.nominal, self.method)
        price_zc = zc.prix()

        put_price = heston_option_price_put(self.r, T, self.protection_barrier * self.sous_jacent.S0,
                                            self.sous_jacent.S0, rho, theta, k, eta, v0,
                                            self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)

        coupon_value = self.coupon_rate * self.nominal * exp(-self.r * T)
        reverse_price = price_zc + coupon_value - (put_price * self.nominal / self.sous_jacent.S0)

        return reverse_price



# Configuration Monte Carlo personnalisée
mc_config = MonteCarloConfig(Nmc=5000, N=200, seed=42)
# Création des options
action = Action("AAPL", 215)
maturite=10
autocall_params = {
'observation_dates': [0.25, 0.5, 0.75, 1],  # tous les 3 mois
'barriers': [1.0, 1.0, 1.0, 1.0],            # barrière 100% S0
'coupons': [0.02, 0.04, 0.06, 0.08],          # coupons cumulés
'protection_barrier': 0.6                    # protection à 60%
}
S0 = 215
r = 0.045
param = [0.117478, -0.353219, 0.063540, 1.251697, 0.898913 ]

Call_test = Call(action,1,param,r,230,mc_config)
Call_test.plot_payoff()

 # Instanciation de l'Autocall orienté objet
autocall = Autocall(
    sous_jacent=action,
    maturite=1,
    parametres=param,
    r=r,
    autocall_params=autocall_params,
    mc_config=mc_config
)

print(f"Le prix de l'autocall est : {autocall.price():.4f}")

# Paramètres réalistes
action = Action("AAPL", 100.0)  # Spot = 100€
maturite_rc = 1
rc = ReverseConvertible(
    sous_jacent=action,
    maturite=maturite_rc,  # 2 ans
    parametres=param,
    r=0.03,  # 3% taux sans risque
    coupon_rate=0.10,  # 10% de coupon
    protection_barrier=0.7,  # Barrière à 70% du spot
    nominal=1000,  # Nominal de 1000€
    mc_config=MonteCarloConfig(Nmc=100000)
)

prix = rc.price()
print(f"Prix de la Reverse Convertible: {prix:.2f}€")
