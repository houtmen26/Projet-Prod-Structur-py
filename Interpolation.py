from abc import ABC, abstractmethod
import numpy as np


class Interpolation(ABC):
    """
    Classe abstraite pour l'interpolation des courbes de taux.
    """
    def __init__(self, maturites, taux):
        """
        :param maturites: list : Liste des maturités en années
        :param taux: list : Liste des taux correspondants
        """
        self.maturites = np.array(maturites)
        self.taux = np.array(taux)

    @abstractmethod
    def interpoler(self, maturite_cible):
        """
        Méthode abstraite pour interpoler un taux donné une maturité cible.
        """
        pass
