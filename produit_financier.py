import pandas as pd
import numpy as np
from abc import ABC,abstractmethod
from Call_Heston import heston_option_price_call,heston_option_price_put
from dataclasses import dataclass
from Modele_Heston import payoff_heston
from math import *
import matplotlib.pyplot as plt
import streamlit as st
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

    def plot_payoff(self, ax=None):
        S_range = np.linspace(0.5 * self.strike, 1.5 * self.strike, 100)
        payoff = np.maximum(S_range - self.strike, 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Call Payoff")
        ax.axvline(self.strike, color='gray', linestyle='--', label='Strike')
        ax.set_title("Payoff d'un Call")
        ax.set_xlabel("Prix du sous-jacent")
        ax.set_ylabel("Payoff à maturité")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)

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

    def plot_payoff(self, ax=None):
        S_range = np.linspace(0.5 * self.strike, 1.5 * self.strike, 100)
        payoff = np.maximum(-S_range + self.strike, 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Put Payoff")
        ax.axvline(self.strike, color='gray', linestyle='--', label='Strike')
        ax.set_title("Payoff d'un Put")
        ax.set_xlabel("Prix du sous-jacent")
        ax.set_ylabel("Payoff à maturité")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)



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

    def plot_payoff(self, ax=None):
        K1, K2 = self.strikes
        S_range = np.linspace(0.5 * K1, 1.5 * K2, 200)
        payoff = np.maximum(S_range - K1, 0) - np.maximum(S_range - K2, 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Call Spread Payoff", color='blue')
        ax.axvline(K1, color='green', linestyle='--', label=f'Strike Bas (K1 = {K1})')
        ax.axvline(K2, color='red', linestyle='--', label=f'Strike Haut (K2 = {K2})')
        ax.set_title("Payoff d'un Call Spread à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)

class Straddle(StrategieOption):
    def price(self):
        v0, rho, theta, k, eta = self.parametres
        K = self.strikes[0]
        call = heston_option_price_call(self.r, self.maturite, K, self.sous_jacent.S0,
                                        rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        put = heston_option_price_put(self.r, self.maturite, K, self.sous_jacent.S0,
                                      rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        return call + put

    def plot_payoff(self, ax=None):
        K = self.strikes[0]
        S_range = np.linspace(0.5 * K, 1.5 * K, 200)
        payoff = np.maximum(S_range - K, 0) + np.maximum(K - S_range, 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Straddle Payoff", color='blue')
        ax.axvline(K, color='gray', linestyle='--', label=f'Strike (K = {K})')
        ax.set_title("Payoff d'un Straddle à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)


class Strangle(StrategieOption):
    def price(self):
        v0, rho, theta, k, eta = self.parametres
        K1, K2 = self.strikes
        call = heston_option_price_call(self.r, self.maturite, K1, self.sous_jacent.S0,
                                        rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        put = heston_option_price_put(self.r, self.maturite, K2, self.sous_jacent.S0,
                                      rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        return call + put

    def plot_payoff(self, ax=None):
        K1, K2 = self.strikes
        S_range = np.linspace(0.5 * K2, 1.5 * K1, 200)
        payoff = np.maximum(S_range - K1, 0) + np.maximum(K2 - S_range, 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Strangle Payoff", color='darkorange')
        ax.axvline(K1, color='green', linestyle='--', label=f'Strike Call (K1 = {K1})')
        ax.axvline(K2, color='red', linestyle='--', label=f'Strike Put (K2 = {K2})')
        ax.set_title("Payoff d'un Strangle à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)


class Strip(StrategieOption):
    def price(self):
        v0, rho, theta, k, eta = self.parametres
        K = self.strikes[0]
        call = heston_option_price_call(self.r, self.maturite, K, self.sous_jacent.S0,
                                        rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        put = heston_option_price_put(self.r, self.maturite, K, self.sous_jacent.S0,
                                      rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        return call + 2 * put

    def plot_payoff(self, ax=None):
        K = self.strikes[0]
        S_range = np.linspace(0.5 * K, 1.5 * K, 200)
        payoff = np.maximum(S_range - K, 0) + 2 * np.maximum(K - S_range, 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Strip Payoff", color='purple')
        ax.axvline(K, color='gray', linestyle='--', label=f'Strike (K = {K})')
        ax.set_title("Payoff d'un Strip à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)


class Strap(StrategieOption):
    def price(self):
        v0, rho, theta, k, eta = self.parametres
        K = self.strikes[0]
        call = heston_option_price_call(self.r, self.maturite, K, self.sous_jacent.S0,
                                        rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        put = heston_option_price_put(self.r, self.maturite, K, self.sous_jacent.S0,
                                      rho, theta, k, eta, v0, self.mc_config.Nmc, self.mc_config.N, self.mc_config.seed)
        return 2 * call + put

    def plot_payoff(self, ax=None):
        K = self.strikes[0]
        S_range = np.linspace(0.5 * K, 1.5 * K, 200)
        payoff = 2 * np.maximum(S_range - K, 0) + np.maximum(K - S_range, 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Strap Payoff", color='teal')
        ax.axvline(K, color='gray', linestyle='--', label=f'Strike (K = {K})')
        ax.set_title("Payoff d'un Strap à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)

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

    def plot_payoff(self, ax=None):
        K = self.strikes[0]
        self.h1 = (0.5 / 100) * self.sous_jacent.S0
        S_range = np.linspace(0.8 * K, 1.2 * K, 500)
        payoff = np.where((S_range >= K) & (S_range <= K + self.h1), 1 / self.h1, 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Option Call Binaire (approximation)")
        ax.axvline(K, color='gray', linestyle='--', label=f'Strike (K = {K})')
        ax.axvline(K + self.h1, color='red', linestyle='--', label=f'K + h1 (h1 = {self.h1:.2f})')
        ax.set_title("Payoff d'une Option Call Binaire à maturité (approximation dérivée)")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff")
        ax.set_ylim(-0.1, 1.5 / self.h1)
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)


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
                 strike, barriere, mc_config: MonteCarloConfig = None):
        super().__init__(sous_jacent, maturite, parametres, r)
        HestonOptionMixin.__init__(self, mc_config)
        self.strike = strike
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

    def plot_payoff(self, ax=None):
        S_range = np.linspace(0.5 * self.barriere, 1.5 * self.barriere, 300)
        payoff = np.where(S_range >= self.barriere, np.maximum(S_range - self.strike, 0), 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Up-and-In Call Payoff", color='blue')
        ax.axvline(self.barriere, color='purple', linestyle='--', label=f'Barrière (H={self.barriere})')
        ax.axvline(self.strike, color='gray', linestyle='--', label=f'Strike (K={self.strike})')
        _, prob_activation = self.price()
        ax.text(0.35, 0.95, f"Proba activation: {prob_activation:.2%}",
                transform=ax.transAxes,
                fontsize=12, color='black', bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
        ax.set_title("Payoff d'une Option Up-and-In Call à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)

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

    def plot_payoff(self, ax=None):
        S_range = np.linspace(0.5 * self.barriere, 1.5 * self.barriere, 300)
        payoff = np.where(S_range < self.barriere, np.maximum(S_range - self.strike, 0), 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Up-and-Out Call Payoff", color='red')
        ax.axvline(self.barriere, color='purple', linestyle='--', label=f'Barrière (H={self.barriere})')
        ax.axvline(self.strike, color='gray', linestyle='--', label=f'Strike (K={self.strike})')
        _, prob_activation = self.price()
        ax.text(0.35, 0.95, f"Proba désactivation: {prob_activation:.2%}",
                transform=ax.transAxes,
                fontsize=12, color='black', bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
        ax.set_title("Payoff d'une Option Up-and-Out Call à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)

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

    def plot_payoff(self, ax=None):
        S_range = np.linspace(0.5 * self.strike, 1.5 * self.strike, 300)
        payoff = np.where(S_range > self.barriere, np.maximum(S_range - self.strike, 0), 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Down-and-Out Call Payoff", color='green')
        ax.axvline(self.barriere, color='purple', linestyle='--', label=f'Barrière (H={self.barriere})')
        ax.axvline(self.strike, color='gray', linestyle='--', label=f'Strike (K={self.strike})')
        _, prob_activation = self.price()
        ax.text(0.35, 0.95, f"Proba désactivation: {prob_activation:.2%}",
                transform=ax.transAxes,
                fontsize=12, color='black', bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
        ax.set_title("Payoff d'une Option Down-and-Out Call à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)


class OptionBarriereUpAndInPut(ProduitFinancier, HestonOptionMixin):
    """Option barrière Up-and-In Put. Activée seulement si le sous-jacent dépasse la barrière."""
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

        max_paths = np.max(S_paths, axis=1)
        S_T = S_paths[:, -1]

        payoff = np.maximum(self.strike - S_T, 0)
        payoffs = np.where(max_paths >= self.barriere, payoff, 0)  # Activé si barrière franchie

        discounted = np.exp(-self.r * self.maturite) * payoffs
        prob_activation = np.mean(max_paths >= self.barriere)

        return np.mean(discounted), prob_activation

    def plot_payoff(self, ax=None):
        S_range = np.linspace(0.5 * self.barriere, 1.5 * self.barriere, 300)
        payoff = np.where(S_range >= self.barriere, np.maximum(self.strike - S_range, 0), 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Up-and-In Put Payoff", color='orange')
        ax.axvline(self.barriere, color='purple', linestyle='--', label=f'Barrière (H={self.barriere})')
        ax.axvline(self.strike, color='gray', linestyle='--', label=f'Strike (K={self.strike})')
        _, prob_activation = self.price()
        ax.text(0.35, 0.95, f"Proba activation: {prob_activation:.2%}",
                transform=ax.transAxes,
                fontsize=12, color='black', bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
        ax.set_title("Payoff d'une Option Up-and-In Put à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)

class OptionBarriereUpAndOutPut(ProduitFinancier, HestonOptionMixin):
    """Option barrière Up-and-Out Put. Désactivée si le sous-jacent dépasse la barrière."""
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

        max_paths = np.max(S_paths, axis=1)
        S_T = S_paths[:, -1]

        payoff = np.maximum(self.strike - S_T, 0)
        payoffs = np.where(max_paths >= self.barriere, 0, payoff)  # Désactivé si barrière franchie

        discounted = np.exp(-self.r * self.maturite) * payoffs
        prob_desactivation = np.mean(max_paths >= self.barriere)

        return np.mean(discounted), prob_desactivation

    def plot_payoff(self, ax=None):
        S_range = np.linspace(0.5 * self.barriere, 1.5 * self.barriere, 300)
        payoff = np.where(S_range < self.barriere, np.maximum(self.strike - S_range, 0), 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Up-and-Out Put Payoff", color='teal')
        ax.axvline(self.barriere, color='purple', linestyle='--', label=f'Barrière (H={self.barriere})')
        ax.axvline(self.strike, color='gray', linestyle='--', label=f'Strike (K={self.strike})')
        _, prob_activation = self.price()
        ax.text(0.35, 0.95, f"Proba désactivation: {prob_activation:.2%}",
                transform=ax.transAxes,
                fontsize=12, color='black', bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
        ax.set_title("Payoff d'une Option Up-and-Out Put à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)

class NoteCapitalProtegee(ProduitFinancier, HestonOptionMixin):
    def __init__(self, sous_jacent: Action, maturite, parametres, r,
                 strike, barriere, rebate=0, nominal=1000, method="continu",
                 mc_config: MonteCarloConfig = None):
        super().__init__(sous_jacent, maturite, parametres, r)
        HestonOptionMixin.__init__(self, mc_config)
        self.strike = strike
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

    def plot_payoff(self, ax=None):
        S_range = np.linspace(0.5 * self.strike, 1.5 * self.barriere, 300)
        payoff_call = np.maximum(S_range - self.strike, 0) * (self.nominal / self.sous_jacent.S0)
        payoff = np.where(S_range < self.barriere, self.nominal + payoff_call, self.nominal + self.rebate)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Payoff Note Capital Protégée", color='blue')
        ax.axhline(self.nominal, color='gray', linestyle='--', label=f'Capital Protégé ({self.nominal}€)')
        ax.axvline(self.strike, color='green', linestyle='--', label=f'Strike (K={self.strike})')
        ax.axvline(self.barriere, color='purple', linestyle='--', label=f'Barrière KO (H={self.barriere})')

        # Ajout d'informations sur le rebate
        if self.rebate > 0:
            ax.axhline(self.nominal + self.rebate, color='red', linestyle=':',
                       label=f'Rebate ({self.rebate:.2f}€)')
            ax.text(0.7, 0.85, f"Rebate: {self.rebate:.2f}€",
                    transform=ax.transAxes, color='red')

        ax.set_title("Payoff d'une Note à Capital Protégé à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)

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

    def plot_payoff(self, ax=None):
        S0 = self.sous_jacent.S0
        K_barrier = self.protection_barrier * S0
        coupon = self.coupon_rate * self.nominal
        S_range = np.linspace(0.5 * K_barrier, 1.5 * S0, 300)
        payoff = np.where(
            S_range >= K_barrier,
            self.nominal + coupon,
            S_range * (self.nominal / S0) + coupon
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Reverse Convertible Payoff", color='navy', linewidth=2)
        ax.axvline(K_barrier, linestyle='--', color='purple', label=f'Barrière (H = {K_barrier:.2f})')
        ax.axhline(self.nominal + coupon, linestyle='--', color='gray',
                   label=f'Nominal + Coupon ({self.nominal + coupon:.0f}€)')
        ax.set_title("Payoff d'un Reverse Convertible à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)


class OptionBarriereDownAndInCall(ProduitFinancier, HestonOptionMixin):
    """Option barrière Down-and-In Call. Activée seulement si le sous-jacent tombe sous la barrière."""

    def __init__(self, sous_jacent: Action, maturite: float, parametres: list,
                 r: float, strike: float, barriere: float, mc_config: MonteCarloConfig = None):
        """
        Args:
            strike: Prix d'exercice (K)
            barriere: Niveau d'activation (H < S0)
        """
        super().__init__(sous_jacent, maturite, parametres, r)
        HestonOptionMixin.__init__(self, mc_config)
        self.strike = strike
        self.barriere = barriere

    def price(self) :
        """Calcule le prix et la probabilité d'activation.

        Returns:
            tuple: (prix_estime, probabilite_activation)
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

        # Vérification de la barrière
        min_paths = np.min(S_paths, axis=1)
        S_T = S_paths[:, -1]

        # Calcul du payoff
        payoff_call = np.maximum(S_T - self.strike, 0)
        payoffs = np.where(min_paths <= self.barriere, payoff_call, 0)  # Activé si barrière franchie

        # Actualisation et calculs finaux
        discounted_payoffs = np.exp(-self.r * self.maturite) * payoffs
        prob_activation = np.mean(min_paths <= self.barriere)
        price = np.mean(discounted_payoffs)

        return price, prob_activation

    def plot_payoff(self, ax=None):
        S_range = np.linspace(0.5 * self.barriere, 1.5 * self.strike, 300)
        payoff = np.maximum(S_range - self.strike, 0)
        payoff = np.where(S_range >= self.strike, payoff, payoff)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(S_range, payoff, label="Down-and-In Call Payoff", color='darkgreen')
        ax.axvline(self.barriere, color='purple', linestyle='--', label=f'Barrière (H={self.barriere})')
        ax.axvline(self.strike, color='gray', linestyle='--', label=f'Strike (K={self.strike})')
        _, prob_activation = self.price()
        ax.text(0.35, 0.95, f"Proba activation: {prob_activation:.2%}",
                transform=ax.transAxes,
                fontsize=12, color='black', bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
        ax.set_title("Payoff d'une Option Down-and-In Call à maturité")
        ax.set_xlabel("Prix du sous-jacent $S_T$")
        ax.set_ylabel("Payoff (€)")
        ax.grid(True)
        ax.legend()

        if ax is None:
            st.pyplot(fig)



