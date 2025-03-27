import numpy as np
from Interpolation import Interpolation

class InterpolationLineaire(Interpolation):
    def __init__(self, maturites, taux):
        super().__init__(maturites, taux)

        
    def interpoler(self, maturite_cible):
        """
        Interpolation linéaire à la main.
        """
        # Si la maturité cible est hors bornes, on extrapole avec les valeurs les plus proches
        if maturite_cible <= self.maturites[0]:
            return self.taux[0]
        elif maturite_cible >= self.maturites[-1]:
            return self.taux[-1]

        # Trouver les deux points encadrant maturite_cible
        for i in range(len(self.maturites) - 1):
            if self.maturites[i] <= maturite_cible <= self.maturites[i + 1]:
                x1, x2 = self.maturites[i], self.maturites[i + 1]
                y1, y2 = self.taux[i], self.taux[i + 1]
                return y1 + (y2 - y1) * (maturite_cible - x1) / (x2 - x1)

        raise ValueError("Erreur d'interpolation linéaire.")
