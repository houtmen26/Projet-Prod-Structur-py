from datetime import datetime
from math import exp
from Produit import Produit  
from Maturite import Maturite

class ZeroCoupon(Produit):
    def __init__(self, nom: str, taux: float, maturite: Maturite, nominal: float, methode: str):
        """
        Initialisation du Zero Coupon.

        :param nom: str : Le nom du produit.
        :param taux: float : Le taux d'intérêt (ex: 0.03 pour 3%).
        :param maturite: Maturite : Un objet de la classe Maturite pour gérer la durée et la convention.
        :param nominal: float : Le nominal du produit.
        :param methode: str : La méthode d actualisation ('lineaire', 'actuariel', 'continu').
        """
        super().__init__(nom)  # Appel du constructeur de Produit
        self.taux = taux
        self.maturite = maturite
        self.nominal = nominal
        self.methode = methode.lower()

    def prix(self) -> float:
        """
        Calcule le prix du zéro coupon en fonction de la méthode choisie.

        :return: float : La valeur actuelle du zéro coupon.
        """
        T = self.maturite.maturite_en_annees  # Récupérer la maturité selon la convention choisie

        if self.methode == "lineaire":
            prix_unitaire = 1 / (1 + self.taux * T)
        elif self.methode == "actuariel":
            prix_unitaire = 1 / (1 + self.taux) ** T
        elif self.methode == "continu":
            prix_unitaire = exp(-self.taux * T)
        else:
            raise ValueError("Méthode non reconnue. Choisir entre 'lineaire', 'actuariel', 'continu'.")

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
