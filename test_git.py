# ceci est un test

from Maturite import Maturite
from Action import Action
from datetime import datetime
import pandas as pd
from Taux import Taux
from Call import Call
from Put import Put
from math import exp
from ZeroCoupon import ZeroCoupon

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

# Création d'une option call avec les paramètres spécifiés
call_option = Call(sous_jacent=action, 
                   nom="Call sur AAPL", 
                   sens="long",
                   nominal=1, 
                   strike=150, 
                   volatilite=0.20, 
                   taux_sans_risque=0.02, 
                   maturite=1, 
                   date_valo=datetime(2024, 3, 20))

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
                 taux_sans_risque=0.02, 
                 maturite=1, 
                 date_valo=datetime(2024, 3, 20))

# Calcul du prix de l'option à la date de valorisation
print(f"Prix de l'option Put : {put_option.prix()}")  # Affiche le prix de l'option

# Description de l'option Put
print(put_option)  # Affiche les détails de l'option put


# Vérification de la parité Call-Put
C=call_option.prix()
P=put_option.prix()
S0=action.prix(datetime(2024, 3, 20))
K=150
r=0.02
T=1.0
# Calcul de la parité Call-Put
lhs = C - P  # C - P
rhs = S0 - K * exp(-r * T)  # S0 - K * exp(-rT)

# Comparaison des résultats
if abs(lhs - rhs) < 1e-4:
    print("La parité Call-Put est respectée.")
else:
    print("La parité Call-Put n'est pas respectée.")


# Création d'un objet Maturite avec convention Act/360
maturite = Maturite(val_date="2024-03-25", mat_date="2026-03-25", convention="Act/360")

# Création d’un Zero Coupon avec un taux de 3% et un nominal de 1000
zc = ZeroCoupon(nom="Zero Coupon 2 ans", taux=0.03, maturite=maturite, nominal=1000, methode="actuariel")

# Affichage du prix et de la description
print(zc.prix())  # Affiche uniquement le prix du Zero Coupon
print(zc)  # Affiche une description détaillée
