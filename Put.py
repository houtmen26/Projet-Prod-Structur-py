from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime
from ProduitDerive import ProduitDerive
from Taux import Taux
from Action import Action
from Produit import Produit
from Maturite import Maturite
from CourbeZeroCoupon import CourbeZeroCoupon

class Put(ProduitDerive):
    def __init__(self, sous_jacent: Produit, nom: str, sens: str, nominal: float, strike: float,
                 maturite: Maturite, date_valo: datetime, courbe_zc: CourbeZeroCoupon,
                 volatilite: float = None, methode_interpolation="cubique"):
        """
        Initialisation d'un Put.

        :param sous_jacent: Produit : Le sous-jacent (instance d'Action, Taux, etc.)
        :param nom: str : Le nom du produit dérivé
        :param sens: str : 'long' ou 'short'
        :param nominal: float : nombre de contrat dans ce cas
        :param strike: float : Le prix d'exercice
        :param volatilite: float : La volatilité annuelle (en pourcentage)
        :param maturite: Maturite : La maturité de l'option (en années)
        :param date_valo: datetime : La date de valorisation
        """
        super().__init__(sous_jacent, nom, sens, nominal, maturite, courbe_zc, 
                         methode_interpolation)
        self.strike = strike  # Le prix d'exercice
        self.volatilite = volatilite
        self.date_valo = date_valo  # Date de valorisation

    def prix(self, date: datetime = None):
        """
        Calcul du prix du produit dérivé (Call) selon le type de sous-jacent.
        Si le sous-jacent est un taux, on calcule le prix comme un caplet.
        Si le sous-jacent est une action, on utilise la formule de Black-Scholes.

        :param date: datetime : La date de valorisation. Si non fournie, utilise la date de valorisation par défaut.
        :return: float : Le prix de l'option.
        """
        if isinstance(self.sous_jacent, Taux):  # Si sous-jacent est un produit de taux, utiliser le caplet
            return self.prix_caplet(date)
        elif isinstance(self.sous_jacent, Action):  # Si sous-jacent est une action, utiliser Black-Scholes
            return self.prix_black_scholes(date)
        else:
            raise ValueError("Le type de sous-jacent n'est pas supporté pour ce produit dérivé.")

    def prix_black_scholes(self, date: datetime = None):
        """
        Calcul du prix du Call sur une Action basé sur la formule de Black-Scholes.

        :param date: datetime : La date de valorisation.
        :return: float : Le prix de l'option Call.
        """
        if date is None:
            date = self.date_valo
        
        # Récupérer le prix du sous-jacent à la date de valorisation
        S0 = self.sous_jacent.prix(date)  # Prix initial du sous-jacent
        
        # Si la volatilité et/ou le taux sans risque ne sont pas renseignés, appel à la fonction de calibration
        if self.volatilite is None:
            self.calibrer_parametres()  # Fonction à définir plus tard
        
        # Récupérer le taux sans risque de la courbe ZC pour la maturité
        taux_sans_risque = self.interpolation.interpoler(self.maturite)

        # Calcul des paramètres de Black-Scholes
        d1 = (log(S0 / self.strike) + (taux_sans_risque + 0.5 * self.volatilite ** 2) * self.maturite) / (self.volatilite * sqrt(self.maturite))
        d2 = d1 - self.volatilite * sqrt(self.maturite)
        
        # Calcul du prix de l'option call avec la formule de Black-Scholes
        prix_put = self.strike * exp(-taux_sans_risque * self.maturite) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        return prix_put * self.nominal

    def prix_caplet(self, date: datetime = None):
        """
        Calcul du prix du Caplet pour un produit de taux d'intérêt selon la formule de Black-Scholes.
        A REVOIR : charger la courbe de taux et calculer le forward de taux

        :param date: datetime : La date de valorisation.
        :return: float : Le prix du caplet.
        """

        pass

    def calibrer_parametres(self):
        """
        Fonction pour calibrer la volatilité si nécessaire.
        Cette fonction doit être implémentée pour ajuster les paramètres si ceux-ci ne sont pas précisés.
        """
        # Squelette de la fonction de calibration : utiliser des modèles ou des données historiques
        pass

    def description(self):
        """
        Retourne une description du produit dérivé (Call ou Caplet).
        """
        return f"Option Put: {self.nom}, Strike: {self.strike}, Volatilité: {self.volatilite*100:.2f}%, " \
               f", Maturité: {self.maturite} ans, " \
               f"Nombre de contrats: {self.nominal}"

    def __str__(self):
        return self.description()
