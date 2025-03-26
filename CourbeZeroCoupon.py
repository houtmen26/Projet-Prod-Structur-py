import matplotlib.pyplot as plt
from math import exp, log
from InterpolationCubique import InterpolationCubique
from InterpolationLineaire import InterpolationLineaire
from InterpolationNelsonSiegel import InterpolationNelsonSiegel

class CourbeZeroCoupon:
    def __init__(self, maturites, taux_swap, convention="continue", type_interpolation: str = "cubique"):
        """
        Initialise la courbe zéro-coupon avec le bootstrapping.
        
        :param maturites: list : Liste des maturités des taux swap (en années).
        :param taux_swap: list : Liste des taux swap correspondants (en décimal).
        :param convention: str : Convention de calcul ("actuariel" ou "continue").
        :param conventype_interpolationtion: str : Interpolation : linéaire, cubique ou Nelson Siegel.
        """
        self.maturites = maturites
        self.taux_swap = taux_swap
        self.convention = convention
        self.type_interpolation = type_interpolation
        self.df = {}  # Stocke les facteurs d'actualisation (DF)
        self.taux_zero_coupon = self.bootstrapper()  # On construit la courbe ZC

    def bootstrapper(self):
        """
        Calcule la courbe zéro-coupon par bootstrapping.
        """
        # initialisation de l'interpolation
        if self.type_interpolation == "lineaire":
            interpolation = InterpolationLineaire(self.maturites, self.taux_swap)
        elif self.type_interpolation == "cubique":
            interpolation = InterpolationCubique(self.maturites, self.taux_swap)
        elif self.type_interpolation == "nelson_siegel":
            interpolation = InterpolationNelsonSiegel(self.maturites, self.taux_swap)  
        else:
            raise ValueError("Type d'interpolation non reconnu. Choisissez parmi : 'lineaire', 'cubique' ou 'nelson_siegel'.")



        taux_zc = {}  # Liste des taux ZC calculés
        index_1 = self.maturites.index(1)  # Trouve l'index où maturite == 1
        for i in range(index_1 + 1):
            T = self.maturites[i]
            S = self.taux_swap[i]  # Taux swap pour la maturité T
            if T==1:
                self.df[T] = 1 / (1 + S)
            taux_zc[T] = S

        for T in range(2, max(self.maturites) + 1):
            # S si présent sinon on interpole
            if T in self.maturites:
                index_T = self.maturites.index(T)
                S = self.taux_swap[index_T]
            else:
                S = interpolation.interpoler(T)

            # Calcul du DF(T) par récurrence
            somme_DF = sum(self.df[j] for j in range(1, T)) #somme des DF précédents
            self.df[T] = (1 - S * somme_DF) / (1 + S)

            if self.convention == "actuariel":
                zc_ti = exp(-log(self.df[T])/T) - 1
            else:  # Convention continue
                zc_ti = -log(self.df[T])/T
            taux_zc[T] = zc_ti
    
        return taux_zc


    def afficher_courbe(self):
        """
        Affiche la courbe des taux zéro-coupon.
        """
        print(f"Courbe des taux ZC ({self.convention}):")
        for T in self.taux_zero_coupon.keys():
            print(f"Maturité {T} ans : {self.taux_zero_coupon[T] * 100:.2f}%")


    def tracer_courbe(self):
        """
        Trace uniquement la courbe des taux zéro-coupon.
        """
        plt.figure(figsize=(10, 6))

        # Courbe des taux zéro-coupon
        maturites_zc = list(self.taux_zero_coupon.keys())
        taux_zc = list(self.taux_zero_coupon.values())
        plt.plot(maturites_zc, taux_zc, label="Courbe Zéro-Coupon", color='blue', linestyle='-')

        # Labels et légende
        plt.xlabel("Maturité (années)")
        plt.ylabel("Taux (%)")
        plt.title(f"Courbe des taux ZC ({self.convention})")
        plt.legend()
        plt.grid(True)
        plt.show()

