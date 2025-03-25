import numpy as np
from Interpolation import Interpolation
from scipy.optimize import least_squares

class InterpolationNelsonSiegel(Interpolation):
    def __init__(self, maturites, taux):
        super().__init__(maturites, taux)
        self.beta0, self.beta1, self.beta2, self.lambd = self.calibrer_modele()


import numpy as np
from Interpolation import Interpolation
from scipy.optimize import least_squares

class InterpolationNelsonSiegel(Interpolation):
    def __init__(self, maturites, taux):
        super().__init__(maturites, taux)
        self.beta0, self.beta1, self.beta2, self.lambd = self.calibrer_nelson_siegel(maturites, taux)

    def fonction_nelson_siegel(self, tau, beta0, beta1, beta2, lambd):
        """
        Fonction de Nelson-Siegel.
        """
        if tau == 0:
            return beta0  # Évite la division par zéro
        return beta0 + beta1 * (1 - np.exp(-tau / lambd)) / (tau / lambd) + beta2 * ((1 - np.exp(-tau / lambd)) / (tau / lambd) - np.exp(-tau / lambd))

    def erreur_nelson_siegel(self, params, maturites, taux_observes):
        """
        Fonction d'erreur entre les taux observés et les taux estimés par le modèle Nelson-Siegel.
        """
        beta0, beta1, beta2, lambd = params
        erreurs = []
        for i in range(len(maturites)):
            taux_model = self.fonction_nelson_siegel(maturites[i], beta0, beta1, beta2, lambd)
            erreurs.append(taux_observes[i] - taux_model)
        return np.array(erreurs)

    def calibrer_nelson_siegel(self, maturites, taux_observes):
        """
        Calibre les paramètres du modèle Nelson-Siegel en minimisant l'erreur quadratique.
        """
        # Estimations initiales des paramètres
        beta0_init = taux_observes[-1]  # Le taux long terme est souvent proche du dernier taux observé
        beta1_init = taux_observes[0] - beta0_init  # Différence entre court et long terme
        beta2_init = 0.02  # Une valeur standard initiale
        lambd_init = 2.0  # Valeur raisonnable

        params_init = [beta0_init, beta1_init, beta2_init, lambd_init]

        # Optimisation avec Levenberg-Marquardt (sans method='lm' car SciPy choisit le meilleur)
        resultat = least_squares(self.erreur_nelson_siegel, params_init, args=(maturites, taux_observes))

        # Extraction des paramètres optimisés
        beta0_opt, beta1_opt, beta2_opt, lambd_opt = resultat.x
        return beta0_opt, beta1_opt, beta2_opt, lambd_opt

    def interpoler(self, maturite_cible):
        """
        Interpolation du taux pour une maturité cible avec les paramètres calibrés.
        """
        return self.fonction_nelson_siegel(maturite_cible, self.beta0, self.beta1, self.beta2, self.lambd)
