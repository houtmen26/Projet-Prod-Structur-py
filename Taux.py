
from datetime import datetime
from Produit import Produit

class Taux(Produit):
    def __init__(self, nom: str, date: datetime, maturite: float, taux: float):
        super().__init__(nom)
        self.date = date
        self.maturite = maturite  # En années (ex: 0.25 pour 3 mois)
        self.taux = taux  # En pourcentage (ex: 3.20 pour 3.20%)
    
    def prix(self):
        """
        Retourne le taux d'intérêt
        """
        return self.taux
    
    def description(self):
        return f"Taux {self.nom}: {100*self.taux:.2f}% à la date {self.date.strftime('%Y-%m-%d')} avec une maturité de {self.maturite} an(s)"
