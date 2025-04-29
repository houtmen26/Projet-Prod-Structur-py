import streamlit as st
import pandas as pd

# Pour lancer l'affichage il faut faire cela dans le terminal , juste run le code ne marchera pas
# streamlit run /Users/houtmenelmorabit/Desktop/GitHub/Projet-Prod-Structur-py/Affichage.py


from produit_financier import *  # Ton module avec toutes les classes

st.title("🧮 Pricing de Produits Structurés")

produit = st.selectbox("Choisissez un produit :", [
    "Call", "Put", "CallSpread", "Straddle", "Strangle", "Strip", "Strap", "OptionCallBinaire", "ReverseConvertible", "Autocall"])

# Paramètres communs
S0 = st.number_input("Prix spot (S0)", value=100.0)
maturite = st.number_input("Maturité (en années)", value=1.0)
r = st.number_input("Taux sans risque (r)", value=0.03)

# Paramètres de Heston
st.subheader("Paramètres du modèle de Heston")
v0 = st.number_input("v0", value=0.04)
rho = st.number_input("rho", value=-0.7)
theta = st.number_input("theta", value=0.04)
k = st.number_input("k", value=1.0)
eta = st.number_input("eta", value=0.2)
param = [v0, rho, theta, k, eta]

# Monte Carlo
Nmc = st.slider("Nombre de simulations Monte Carlo", 1000, 100000, 10000, step=1000)
N = st.slider("Nombre de pas", 50, 500, 200, step=10)

mc_config = MonteCarloConfig(Nmc=Nmc, N=N, seed=42)
action = Action("Asset", S0)

if produit == "Call" or produit == "Put":
    strike = st.number_input("Strike", value=100.0)
    if produit == "Call":
        instrument = Call(action, maturite, param, r, strike, mc_config)
    else:
        instrument = Put(action, maturite, param, r, strike, mc_config)

elif produit in ["CallSpread", "Strangle"]:
    K1 = st.number_input("Strike bas", value=90.0)
    K2 = st.number_input("Strike haut", value=110.0)
    cls = CallSpread if produit == "CallSpread" else Strangle
    instrument = cls(action, maturite, param, r, [K1, K2], mc_config)

elif produit in ["Straddle", "Strip", "Strap"]:
    strike = st.number_input("Strike", value=100.0)
    cls = eval(produit)
    instrument = cls(action, maturite, param, r, [strike], mc_config)

elif produit == "OptionCallBinaire":
    strike = st.number_input("Strike", value=100.0)
    instrument = OptionCallBinaire(action, maturite, param, r, strike, 0.5, mc_config)

elif produit == "ReverseConvertible":
    coupon = st.number_input("Coupon (%)", value=0.1)
    barrier = st.number_input("Barrière de protection (% S0)", value=0.7)
    nominal = st.number_input("Nominal", value=1000)
    instrument = ReverseConvertible(action, maturite, param, r, coupon, barrier, nominal, mc_config=mc_config)

elif produit == "Autocall":
    st.subheader("Paramètres Autocall")
    nb_obs = st.slider("Nombre de dates d'observation", 1, 6, 4)
    observation_dates = [round(i * maturite / nb_obs, 2) for i in range(1, nb_obs + 1)]
    barrier = st.number_input("Barrière (%)", value=1.0)
    coupons = [round((i + 1) * 0.02, 2) for i in range(nb_obs)]
    protection_barrier = st.number_input("Barrière de protection (%)", value=0.6)
    autocall_params = {
        'observation_dates': observation_dates,
        'barriers': [barrier] * nb_obs,
        'coupons': coupons,
        'protection_barrier': protection_barrier
    }
    instrument = Autocall(action, maturite, param, r, autocall_params, mc_config)

if st.button("Calculer le prix"):
    prix = instrument.price()
    st.success(f"💰 Prix estimé : {prix:.4f} €")


