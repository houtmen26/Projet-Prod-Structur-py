from datetime import datetime
from Produit import Produit
import numpy as np
import pandas as pd

class Action(Produit):
    def __init__(self, nom: str, historique: pd.DataFrame):
        super().__init__(nom)
        if not {'Date', 'Prix'}.issubset(historique.columns):
            raise ValueError("Le DataFrame doit contenir les colonnes 'Date' et 'Prix'")
        self.historique = historique.sort_values(by='Date').set_index('Date')
    
    def prix(self, date: datetime = None):
        if date is None:
            return self.historique['Prix'].iloc[-1]  # Dernière valeur
        return self.historique['Prix'].get(date, None)  # Prix à la date donnée ou None si non trouvé
    
    def rendements_journaliers(self):
        """
        Calcule les rendements journaliers de l'action.
        """
        return self.historique['Prix'].pct_change().dropna()
    
    def rendements_cumules(self):
        """
        Calcule les rendements cumulés de l'action.
        """
        return (1 + self.rendements_journaliers()).cumprod() - 1
    
    def volatilite_annuelle(self):
        """
        Calcule la volatilité annualisée de l'action.
        """
        rendements = self.rendements_journaliers()
        return np.std(rendements) * np.sqrt(252)  # 252 jours de bourse par an
    
    def rendement_annuel_moyen(self):
        """
        Calcule le rendement moyen annualisé de l'action.
        """
        rendements = self.rendements_journaliers()
        return np.mean(rendements) * 252