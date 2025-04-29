from datetime import datetime
from math import exp
from Produit import Produit  
from Maturite import Maturite

class ZeroCoupon:
    def __init__(self, nom: str, taux: float, maturite: float, nominal: float, methode: str):
        self.nom = nom
        self.taux = taux
        self.maturite = maturite  # directement un float
        self.nominal = nominal
        self.methode = methode.lower()

    def prix(self) -> float:
        T = self.maturite  # plus d'objet .maturite_en_annees

        if self.methode == "lineaire":
            prix_unitaire = 1 / (1 + self.taux * T)
        elif self.methode == "actuariel":
            prix_unitaire = 1 / (1 + self.taux) ** T
        elif self.methode == "continu":
            prix_unitaire = exp(-self.taux * T)
        else:
            raise ValueError("Méthode non reconnue. Choisir entre 'lineaire', 'actuariel', 'continu'.")

        return prix_unitaire * self.nominal

        return prix_unitaire * self.nominal  # Prix ajusté par le nominal

    def description(self):
        """
        Retourne une description complète du Zero Coupon.
        """
        return (f"Zero Coupon - {self.nom}, Taux: {self.taux:.2%}, "
                f"Méthode: {self.methode}, Prix: {self.prix():.4f}, "
                f"Maturité: {self.maturite.maturite_en_annees:.2f} ans ({self.maturite.convention})")

    def __str__(self):
        return self.description()
