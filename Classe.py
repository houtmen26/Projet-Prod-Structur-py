import numpy as np
import pandas as pd
from datetime import datetime
import scipy
from math import *

class Maturite :
    def __init__(self,val_date:str,mat_date:str,convention : str):
        # Pour convention je vais mettre des fichiers qui seront lié a la convention (il en faut 4)
        self.val_date = datetime.strptime(val_date, "%Y-%m-%d")
        self.mat_date = datetime.strptime(mat_date, "%Y-%m-%d")
        self.convention = convention
        self.maturite_en_annees = self.calculer_maturite()

    def calculer_maturite(self) -> float:
        """
        Calcule la maturité en années selon la convention de jours.

        :return: Maturité en années
        """
        jours = (self.mat_date - self.val_date).days

        if self.convention == "Act/365":
            return jours / 365
        elif self.convention == "Act/360":
            return jours / 360
        elif self.convention == "30/360":
            return ((self.mat_date.year - self.val_date.year) * 360 +
                    (self.mat_date.month - self.val_date.month) * 30 +
                    (self.mat_date.day - self.val_date.day)) / 360
        else:
            raise ValueError("Convention de jours non reconnue. Utiliser 'Act/365', 'Act/360' ou '30/360'.")

    def __str__(self):
        """
        Affiche les informations sur la maturité.
        """
        return (f"Maturité:\n"
                f" - Date de valorisation : {self.val_date.strftime('%Y-%m-%d')}\n"
                f" - Date de maturité : {self.mat_date.strftime('%Y-%m-%d')}\n"
                f" - Maturité en années : {self.maturite_en_annees:.4f} ({self.convention})")


class taux :
    def __init__(self,taux,type_taux,maturite,nominal,frequence):
        self.type = type_taux
        self.taux = taux
        self.matu = maturite
        self.nominal = nominal
        self.compo = frequence
    # obj : Courbe de taxux

    # avec des methodes d'interpo (voir debut)
    def interpo(self):
        scipy.interpolate.CubicSpline()
    # obj : Courbe de taxux
    # avec des methodes d'interpo (voir debut)
    def calcul_taux(self) -> float:
        if self.type== "composé":
            return self.nominal*(1+self.taux/self.compo)**(self.compo*self.matu)
        if self.type == "continu":
            return self.nominal*exp(self.taux*self.matu)



class produits :
    def __init__(self):
        pass
    # obj : pricer un ZC Bond
    # obj2 : pricer un Fixed Rate bond




