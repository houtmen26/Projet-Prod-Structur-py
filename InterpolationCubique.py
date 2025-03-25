import numpy as np
from Interpolation import Interpolation

class InterpolationCubique(Interpolation):

    def __init__(self, maturites, taux):
        super().__init__(maturites, taux)

        
    def interpoler(self, maturite_cible):
        """
        Interpolation cubique manuelle par polynôme de Lagrange.
        """
        if maturite_cible <= self.maturites[0]:
            return self.taux[0]
        elif maturite_cible >= self.maturites[-1]:
            return self.taux[-1]

        # Trouver les 4 points les plus proches
        indices = np.searchsorted(self.maturites, maturite_cible)
        indices = max(1, min(indices, len(self.maturites) - 2))  # Éviter les bords
        x_vals = self.maturites[indices - 1:indices + 3]
        y_vals = self.taux[indices - 1:indices + 3]

        # Calcul du polynôme de Lagrange
        def lagrange(x, x_vals, y_vals):
            result = 0
            for i in range(len(x_vals)):
                term = y_vals[i]
                for j in range(len(x_vals)):
                    if i != j:
                        term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
                result += term
            return result

        return lagrange(maturite_cible, x_vals, y_vals)
