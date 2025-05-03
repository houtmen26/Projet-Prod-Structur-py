from Call_Heston import heston_option_price_put, heston_option_price_call, param
from math import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from produit_financier import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class GreekAnalyzer:
    """Analyse précise des Grecs avec visualisation professionnelle"""

    def __init__(self, product, h=0.005, seed=42):
        self.product = product
        self.h = h  # Petit pas pour différences finies
        self.seed = seed  # Seed fixe pour reproductibilité

    def compute_greeks(self, spot_range=None):
        """Calcule tous les Grecs sur une plage de spots"""
        if spot_range is None:
            S0 = self.product.sous_jacent.S0
            spot_range = np.linspace(0.5 * S0, 1.5 * S0, 50)

        results = {
            'spots': spot_range,
            'delta': [],
            'gamma': [],
            'vega': [],
            'theta': [],
            'rho': []
        }

        original_spot = self.product.sous_jacent.S0
        original_params = self.product.parametres.copy()

        for spot in spot_range:
            self.product.sous_jacent.S0 = spot

            # Delta et Gamma
            delta = self._delta()
            gamma = self._gamma()

            # Vega (réinitialise les paramètres)
            self.product.parametres = original_params.copy()
            vega = self._vega()

            # Theta et Rho
            theta = self._theta()
            rho = self._rho()

            results['delta'].append(delta)
            results['gamma'].append(gamma)
            results['vega'].append(vega)
            results['theta'].append(theta)
            results['rho'].append(rho)

        # Réinitialisation
        self.product.sous_jacent.S0 = original_spot
        self.product.parametres = original_params

        return results

    def _delta(self):
        S0 = self.product.sous_jacent.S0
        h = self.h * S0  # Relative bump size
        f = lambda x: self._price_with_spot(x, self.seed)
        return self._central_diff(f, S0, h)

    def _gamma(self):
        S0 = self.product.sous_jacent.S0
        h = self.h * S0  # Relative bump size
        f = lambda x: self._price_with_spot(x, self.seed)
        return self._second_diff(f, S0, h)

    def _vega(self):
        v0 = self.product.parametres[0]  # Extract initial volatility
        h = self.h * v0  # Relative bump size
        f = lambda x: self._price_with_vol(x, self.seed)
        return self._central_diff(f, v0, h)

    def _theta(self):
        T = self.product.maturite
        one_day = 1 / 365  # 1-day decay
        if T <= one_day:
            return 0  # Avoid division errors
        f = lambda x: self._price_with_maturity(x, self.seed)
        return self._forward_diff(f, T, one_day)

    def _rho(self):
        r = self.product.r
        h = self.h * max(0.01, r)  # Ensure minimum bump
        f = lambda x: self._price_with_rate(x, self.seed)
        return self._central_diff(f, r, h)

    def _central_diff(self, f, x, h):
        return (f(x + h) - f(x - h)) / (2 * h)

    def _forward_diff(self, f, x, h):
        return (f(x - h) - f(x)) / h

    def _second_diff(self, f, x, h):
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

    def _price_with_spot(self, new_spot, seed):
        original = self.product.sous_jacent.S0
        self.product.sous_jacent.S0 = new_spot
        if hasattr(self.product, 'mc_config'):
            self.product.mc_config.seed = seed
        price = self._get_price()
        self.product.sous_jacent.S0 = original
        return price

    def _price_with_vol(self, new_vol, seed):
        original = self.product.parametres[0]
        self.product.parametres[0] = new_vol
        if hasattr(self.product, 'mc_config'):
            self.product.mc_config.seed = seed
        price = self._get_price()
        self.product.parametres[0] = original
        return price

    def _price_with_maturity(self, new_T, seed):
        original = self.product.maturite
        self.product.maturite = new_T
        if hasattr(self.product, 'mc_config'):
            self.product.mc_config.seed = seed
        price = self._get_price()
        self.product.maturite = original
        return price

    def _price_with_rate(self, new_r, seed):
        original = self.product.r
        self.product.r = new_r
        if hasattr(self.product, 'mc_config'):
            self.product.mc_config.seed = seed
        price = self._get_price()
        self.product.r = original
        return price

    def _get_price(self):
        """Handle both single price and (price, proba) tuples"""
        price = self.product.price()
        return price[0] if isinstance(price, tuple) else price

    def plot_all_greeks(self, spot_range=None):
        """Visualisation professionnelle des 5 Grecs"""
        results = self.compute_greeks(spot_range)

        plt.figure(figsize=(14, 10))
        plt.suptitle(f"Sensibilités des Grecs - {self.product.__class__.__name__}\n"
                     f"Spot={self.product.sous_jacent.S0} | Strike={getattr(self.product, 'strike', 'N/A')}",
                     y=1.02, fontsize=14)

        greeks = [
            ('delta', 'Δ - Delta [-1,1]', 'blue', (-1, 1)),
            ('gamma', 'Γ - Gamma [0,∞]', 'green', (0, None)),
            ('vega', 'ν - Vega [0,∞]', 'red', (0, None)),
            ('theta', 'Θ - Theta (1j) [-∞,0]', 'purple', (None, 0)),
            ('rho', 'ρ - Rho [-∞,∞]', 'orange', (None, None))
        ]

        for i, (greek, title, color, ylim) in enumerate(greeks, 1):
            ax = plt.subplot(2, 3, i)
            ax.plot(results['spots'], results[greek], color=color, lw=2)

            # Current spot line
            current_spot = self.product.sous_jacent.S0
            ax.axvline(current_spot, color='k', linestyle='--', alpha=0.5)

            # Formatting
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Prix Spot', fontsize=10)
            ax.set_ylabel('Valeur', fontsize=10)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.tight_layout()
        plt.show()

# Configuration avec seed fixe
params = [0.117478, -0.353219, 0.063540, 1.251697, 0.898913]
action = Action("AAPL", 215)
mc_config = MonteCarloConfig(Nmc=10000, N=252, seed=42)  # Seed fixe ici

# Création du produit
call = Call(
    sous_jacent=action,
    maturite=1,
    parametres=params,
    r=0.05,
    strike=230,
    mc_config=mc_config
)

put = Put(
    sous_jacent=action,
    maturite=1,
    parametres=params,
    r=0.05,
    strike=160,
    mc_config=mc_config
)


call_spread = CallSpread(
    sous_jacent=action,
    maturite=1,
    parametres=params,
    r=0.05,
    strikes=[230,260],
    mc_config=mc_config
)
strike = 220
barriere = 240
S0 = 100
r = 0.03
maturite = 1
strike = 230
h1 = 0.5  # 0.5% du strike+
call_ = Put(action,maturite,param,r,strike,mc_config)

analyzer = GreekAnalyzer(call_, h=0.1, seed=42)
analyzer.plot_all_greeks(spot_range=np.linspace(80, 350, 100))
# Analyse
#analyzer = GreekAnalyzer(call, h=0.01, seed=42)

# Visualisation professionnelle
#analyzer.plot_all_greeks(spot_range=np.linspace(150, 300, 50))















