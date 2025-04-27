import streamlit as st
import pandas as pd

# Pour lancer l'affichage il faut faire cela dans le terminal , juste run le code ne marchera pas
# streamlit run /Users/houtmenelmorabit/Desktop/GitHub/Projet-Prod-Structur-py/Affichage.py
st.title("premier test ")
st.write("hello")


st.write("La calibration par le modèle de Heston nous donne les paramètres suivants :")

# Création du DataFrame avec les paramètres
data = {
    "Paramètre": ["v0", "rho", "theta", "k", "eta"],
    "Valeur": [
        "0.101657 ",
        "-0.576699 ",
        "0.056907 ",
        "0.371344 ",
        "0.237523 "
    ],
    "Caractéristiques" : ["(variance initiale)",
    "(corrélation)",
    "(variance long-terme)",
    "(vitesse de retour)",
    "(vol de vol)"]
}

df = pd.DataFrame(data)

# Affichage du tableau dans Streamlit
st.write(df)