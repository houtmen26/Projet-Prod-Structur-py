from abc import ABC, abstractmethod
from datetime import datetime
from Produit import Produit
from Maturite import Maturite
from CourbeZeroCoupon import CourbeZeroCoupon

from InterpolationCubique import InterpolationCubique
from InterpolationLineaire import InterpolationLineaire
from InterpolationNelsonSiegel import InterpolationNelsonSiegel

class ProduitDerive(ABC):
    def __init__(self, sous_jacent: Produit, nom: str, sens: str, nominal: float, maturite: Maturite,
                 courbe_zc: CourbeZeroCoupon, methode_interpolation: str = "cubique"):
        """
        Initialise un produit dérivé.

        :param sous_jacent: Produit : L'objet représentant le sous-jacent, par exemple une instance de la classe Action ou Taux.
        :param nom: str : Le nom du produit dérivé.
        :param sens: str : Le sens de la position, soit 'long' (acheteur) soit 'short' (vendeur).
        :param nominal: float : Le montant notionnel du contrat.
        :param maturite: Maturite : La maturité du produit dérivé, exprimée en années.
        :param courbe_zc: CourbeZeroCoupon : La courbe des taux zéro-coupon utilisée pour l'actualisation.
        :param methode_interpolation: str : La méthode d'interpolation utilisée pour estimer les taux manquants.
                                           Options : 'lineaire', 'cubique', 'nelson_siegel'.
        """
        if sens not in ["long", "short"]:
            raise ValueError("Le sens doit être 'long' ou 'short'.")

        self.sous_jacent = sous_jacent  # Le produit sous-jacent (Action ou Taux).
        self.nom = nom  # Nom du produit dérivé.
        self.sens = sens  # 'long' ou 'short'.
        self.nominal = nominal  # Montant notionnel du contrat.
        self.maturite = maturite.maturite_en_annees  # Conversion de la maturité en années.
        self.courbe_zc = courbe_zc.taux_zero_coupon  # Stockage de la courbe des taux zéro-coupon.

        # Initialisation de l'interpolation en fonction de la méthode choisie.
        if methode_interpolation == "lineaire":
            self.interpolation = InterpolationLineaire(list(courbe_zc.taux_zero_coupon.keys()), list(courbe_zc.taux_zero_coupon.values()))
        elif methode_interpolation == "cubique":
            self.interpolation = InterpolationCubique(list(courbe_zc.taux_zero_coupon.keys()), list(courbe_zc.taux_zero_coupon.values()))
        elif methode_interpolation == "nelson_siegel":
            self.interpolation = InterpolationNelsonSiegel(list(courbe_zc.taux_zero_coupon.keys()), list(courbe_zc.taux_zero_coupon.values()))
        else:
            raise ValueError("Type d'interpolation non reconnu. Choisissez parmi : 'lineaire', 'cubique' ou 'nelson_siegel'.")


    
    @abstractmethod
    def prix(self, date: datetime = None):
        """
        Méthode abstraite pour calculer le prix du produit dérivé.
        Chaque classe fille devra implémenter cette méthode.
        """
        pass
    
    def description(self):
        """
        Retourne une description du produit dérivé.
        """
        return f"Produit Dérivé: {self.nom}, Sens: {self.sens}, Sous-jacent: {self.sous_jacent.__class__.__name__}, Nominal: {self.nominal}"

    def __str__(self):
        return self.description()