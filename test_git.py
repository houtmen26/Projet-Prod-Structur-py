# ceci est un test

from Maturite import Maturite
from Action import Action
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Taux import Taux
from Call import Call
from Put import Put
from math import exp
from ZeroCoupon import ZeroCoupon
from InterpolationCubique import InterpolationCubique
from InterpolationLineaire import InterpolationLineaire
from InterpolationNelsonSiegel import InterpolationNelsonSiegel
from CourbeZeroCoupon import CourbeZeroCoupon
from Future import Future

# Création d'une instance de Maturite
maturite_obj = Maturite("2024-02-20", "2029-02-20", "Act/365")

# Récupération de la maturité
maturite = maturite_obj.maturite_en_annees  # Utilise l'attribut au lieu d'appeler la méthode directement
print(maturite)


# Création d'un DataFrame de prix fictif
data = {
    "Date": [datetime(2024, 3, 20), datetime(2024, 3, 21), datetime(2024, 3, 22), datetime(2024, 3, 23)],
    "Prix": [150.0, 152.5, 149.8, 151.2]
}
df = pd.DataFrame(data)

# Instanciation de l'action avec le DataFrame
action = Action("AAPL", df)

# Test des méthodes
print(action.prix())  # Dernière valeur : 151.2
print(action.prix(datetime(2024, 3, 21)))  # 152.5
print(action.rendements_journaliers())  # Rendements journaliers
print(action.volatilite_annuelle())  # Volatilité annualisée
print(action.rendement_annuel_moyen())  # Rendement annuel moyen


taux = Taux("E3M", datetime(2024, 3, 21), 0.25, 0.032)

print(taux.description())


# === Exemple d'utilisation ===
maturites = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20]  # Maturités en années
taux_swap = [0.015, 0.016, 0.02, 0.025, 0.03, 0.035, 0.038, 0.04, 0.042]  # Taux swap correspondants

courbe_zc = CourbeZeroCoupon(maturites, taux_swap)
courbe_zc.tracer_courbe()
courbe_zc.afficher_courbe()

# Création d'une option call avec les paramètres spécifiés
call_option = Call(sous_jacent=action, 
                   nom="Call sur AAPL", 
                   sens="long",
                   nominal=1, 
                   strike=150, 
                   volatilite=0.20,  
                   maturite=maturite_obj, 
                   date_valo=datetime(2024, 3, 20),
                   courbe_zc=courbe_zc)

# Calcul du prix de l'option à la date de valorisation
print(f"Prix de l'option Call : {call_option.prix()}")  # Affiche le prix de l'option

# Description de l'option Call
print(call_option)  # Affiche les détails de l'option call

# Création d'une option Put avec les paramètres spécifiés
put_option = Put(sous_jacent=action, 
                   nom="Put sur AAPL", 
                   sens="long",
                   nominal=1, 
                   strike=150, 
                   volatilite=0.20, 
                   maturite=maturite_obj, 
                   date_valo=datetime(2024, 3, 20),
                   courbe_zc=courbe_zc)

# Calcul du prix de l'option à la date de valorisation
print(f"Prix de l'option Put : {put_option.prix()}")  # Affiche le prix de l'option

# Description de l'option Put
print(put_option)  # Affiche les détails de l'option put


# Vérification de la parité Call-Put
C=call_option.prix()
P=put_option.prix()
S0=action.prix(datetime(2024, 3, 20))
K=150
T=maturite_obj.maturite_en_annees
interpo_cubique = InterpolationCubique(list(courbe_zc.taux_zero_coupon.keys()), 
                                     list(courbe_zc.taux_zero_coupon.values()))
r = interpo_cubique.interpoler(T)

# Calcul de la parité Call-Put
lhs = C - P  # C - P
rhs = S0 - K * exp(-r * T)  # S0 - K * exp(-rT)

# Comparaison des résultats
if abs(lhs - rhs) < 1e-4:
    print("La parité Call-Put est respectée.")
else:
    print("La parité Call-Put n'est pas respectée.")

# test du future
future_ = Future(action, "Fut test", "long", 1, maturite_obj, courbe_zc)

print(future_)

# Création d'un objet Maturite avec convention Act/360
maturite = Maturite(val_date="2024-03-25", mat_date="2026-03-25", convention="Act/360")

# Création d’un Zero Coupon avec un taux de 3% et un nominal de 1000
zc = ZeroCoupon(nom="Zero Coupon 2 ans", taux=0.03, maturite=maturite, nominal=1000, methode="actuariel")

# Affichage du prix et de la description
print(zc.prix())  # Affiche uniquement le prix du Zero Coupon
print(zc)  # Affiche une description détaillée


maturites = [0.25, 0.5, 1, 2, 5, 10, 20]  # Maturités en années
taux = [0.01, 0.012, 0.015, 0.018, 0.022, 0.025, 0.03]  # Taux en décimal

interp_lin = InterpolationLineaire(maturites, taux)
interp_cub = InterpolationCubique(maturites, taux)
interp_ns = InterpolationNelsonSiegel(maturites, taux)

print("Interpolation linéaire à 3 ans:", interp_lin.interpoler(3))
print("Interpolation cubique à 3 ans:", interp_cub.interpoler(3))
print("Interpolation Nelson-Siegel à 3 ans:", interp_ns.interpoler(3))


# Maturités pour l'affichage des courbes (grille fine)
maturites_fines = np.linspace(min(maturites), max(maturites), 20)

# Calcul des taux interpolés
taux_lin = [interp_lin.interpoler(m) for m in maturites_fines]
taux_cub = [interp_cub.interpoler(m) for m in maturites_fines]
taux_ns = [interp_ns.interpoler(m) for m in maturites_fines]

# Affichage des courbes
plt.figure(figsize=(10, 5))
plt.plot(maturites_fines, taux_lin, label="Interpolation Linéaire", linestyle="dashed")
plt.plot(maturites_fines, taux_cub, label="Interpolation Cubique", linestyle="dotted")
plt.plot(maturites_fines, taux_ns, label="Interpolation Nelson-Siegel", linestyle="solid")
plt.scatter(maturites, taux, color="red", label="Taux observés")  # Points de données

# Ajout des légendes et labels
plt.xlabel("Maturité (années)")
plt.ylabel("Taux (%)")
plt.title("Comparaison des interpolations de la courbe des taux")
plt.legend()
plt.grid()
plt.show()
