from math import exp, log
from Produit import Produit
from Maturite import Maturite
from ProduitDerive import ProduitDerive
from CourbeZeroCoupon import CourbeZeroCoupon
from Action import Action
from Taux import Taux

class Future(ProduitDerive):
    def __init__(self, sous_jacent, nom, sens, nominal, maturite, courbe_zc, 
             methode_interpolation="cubique", convention_future_taux: str = "continue"):
        """
        Initialise un contrat Future.

        :param sous_jacent: objet : Instance de la classe Action ou Taux représentant le sous-jacent du Future.
        :param nom: str : Nom du contrat Future.
        :param sens: str : Le sens de la position, soit 'long' (acheteur) soit 'short' (vendeur).
        :param nominal: float : Le montant notionnel du contrat Future.
        :param maturite: objet : L'objet représentant la maturité du contrat Future, généralement en années.
        :param courbe_zc: objet : L'objet CourbeZeroCoupon, qui fournit la courbe des taux zéro-coupon pour l'actualisation.
        :param methode_interpolation: str : La méthode d'interpolation utilisée pour estimer les taux manquants.
                                            Choisissez parmi : 'lineaire', 'cubique', 'nelson_siegel'. Par défaut, 'cubique'.
        :param convention_future_taux: str : La convention de calcul des taux de l'instrument dérivé.
                                            Par défaut, "continue" qui correspond à la convention généralement utilisée sur les futures.
        """
        super().__init__(sous_jacent, nom, sens, nominal, maturite, courbe_zc, methode_interpolation)
        self.convention_future_taux = convention_future_taux  # Convention des taux futures, par défaut "ACT/360".
        # self.convention_calcul_maturite = maturite.convention()


    def prix(self):
        """
        Calcule le prix du contrat à terme en fonction du sous-jacent.
        """
        if self.maturite not in self.courbe_zc:
            taux_zc = self.interpolation.interpoler(self.maturite)
        else:
            taux_zc = self.courbe_zc[self.maturite]  # Taux zéro-coupon

        if isinstance(self.sous_jacent, Action):
            # F = S0 * exp((r - q) * T) avec r = taux zéro-coupon et q = dividende supposé nul
            prix_future = self.sous_jacent.prix() * exp(taux_zc * self.maturite)
        
        elif isinstance(self.sous_jacent, Taux):
            # Pour un contrat future de taux (ex: Euribor, obligation)
            maturite_taux_eloigne = self.sous_jacent.maturite + self.maturite # pour le calcul Future
            taux_zc_eloigne = self.interpolation.interpoler(maturite_taux_eloigne)
            tx_future = self.future_taux(self.maturite, maturite_taux_eloigne, taux_zc, taux_zc_eloigne)
            prix_future = 100 - tx_future
        else:
            raise ValueError("Type de sous-jacent non reconnu. Choisissez 'action' ou 'taux'.")

        return prix_future
    
    def future_taux(self, T1, T2, ZC_T1, ZC_T2):
        """
        Calcule le taux forward à partir des taux zéro-coupon et de la convention de calcul.

        :param T1: float : La première maturité (T1).
        :param T2: float : La seconde maturité (T2).
        
        :return: float : Le taux forward pour la période [T1, T2].
        """
        # Calcul de la différence d'années entre les deux maturités
        delta_T = T2 - T1

        if self.convention_future_taux == "continue":
            # Calcul du taux forward en convention continue
            taux_forward = (ZC_T2 * T2 - ZC_T1 * T1) / delta_T

        elif self.convention_future_taux == "lineaire":
            # Calcul du taux forward en convention linéaire
            taux_forward = (((1 + T2 * ZC_T2) / (1 + T1*ZC_T1)) - 1) / delta_T
        
        else:
            raise ValueError("Convention future taux non reconnue. Choisissez parmi : 'continue' ou 'lineaire'.")
        
        return taux_forward

    def __str__(self):
        """
        Affiche les informations du contrat future.
        """
        prix = self.prix()
        return (f"Contrat Future sur {self.sous_jacent} :\n"
                f" - Type : {self.sous_jacent}\n"
                f" - Maturité : {self.maturite} ans\n"
                f" - Prix du future : {prix:.4f}")

