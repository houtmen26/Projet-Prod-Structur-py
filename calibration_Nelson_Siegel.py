import numpy as np
import pandas as pd

def nelson_siegel(t,beta0,beta1,beta2,coef,lambda_coeff):
    return beta0 +  beta1*(1-np.exp(-t/lambda_coeff))/(t/lambda_coeff) +beta2*(1-np.exp(-t/lambda_coeff)/(t/lambda_coeff)-np.exp(-t/lambda_coeff))

# Calibration
df = pd.read_csv('RateCurve.csv',sep=";")
def convert_mat(pillar):
    if "M" in pillar :
        return int(pillar.replace("M",""))/12
    if "Y" in pillar :
        return int(pillar.replace("M",""))
    raise ValueError("Unknown format") # Permet de renvoyer un beau msg d'erreur
df["maturity"] = df["Pillar"].map(convert_mat)  # Grosse difference de puissance entre map et Apply (donc tjr uitlier map)
# Methode np.vectorize on lui donne une fct et apres les arguments et np.vectorize va vectoriser la fonction  a toute vitesse
maturities = list(df["maturity"])
rates = list(df["Rate"])

from scipy.optimize import curve_fit
popt,_= curve_fit(nelson_siegel,maturities,rates)




