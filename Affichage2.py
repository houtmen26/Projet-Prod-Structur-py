import streamlit as st
import pandas as pd
import numpy as np
import time
from scipy.optimize import minimize
from Modele_Heston import *
from Call_Heston import *

# Configuration de la page
st.set_page_config(layout="wide", page_title="Pricing Structuré - Heston Calibration")
# Paramètres calibrés par défaut
DEFAULT_PARAMS = {
    'v0': 0.117478,
    'rho': -0.353219,
    'theta': 0.063540,
    'k': 1.251697,
    'eta': 0.898913
}


# ======================================
# SECTION 1 : PAGE D'ACCUEIL
# ======================================
def show_home():
    st.title("🏠 Bienvenue dans l'outil Dauphinance")
    st.markdown("""
    **Cet outil permet de :**
    - Calibrer le modèle de Heston sur vos données marché
    - Pricer différents produits structurés
    - Analyser les sensibilités de vos produits structurés
    """)

    # Images côte à côte
    col1, col2 = st.columns(2)

    with col1:
        st.image("houtmen.JPG", caption="Houtmen El Morabit", width=300)

    with col2:
        st.image("sami.jpeg", caption="Sami Amghar", width=430)

    # Mot du fondateur
    st.markdown("---")
    st.subheader("💬 Mot du fondateur")

    st.markdown("""
    Bienvenue dans cet outil conçu avec passion pour les passionnés de finance quantitative.  
    Notre objectif ? Rendre les modèles avancés comme celui de **Heston** et le pricing accessibles, intuitifs, et performants.  
    N'hésitez pas à explorer, tester, et surtout... apprendre !
    
    *– Houtmen et Sami*
    """)

# ======================================
# SECTION 2 : CALIBRATION HESTON
# ======================================

def show_calibration():
    st.title("🔧 Calibration du Modèle Heston")

    st.markdown(
        "Cette section permet de calibrer le modèle de Heston sur des options Apple et de fournir les cinq paramètres du modèle, qui nous seront très utiles pour la suite.")
    st.markdown(" Voici le temps de calibration indicatif par nombre de simulation de Monte Carlo ")
    data = {
        "100 Simulation": ["1 minute"],
        "1 000 Simulations": [ "7 minutes"],
        "10 000 Simulations": [ "45 minutes"]
    }

    df = pd.DataFrame(data)

    st.table(df)

    # Initialisation des paramètres dans session_state
    if 'params' not in st.session_state:
        st.session_state.params = DEFAULT_PARAMS
        st.session_state.calibrated = False  # False car ce sont des valeurs par défaut
    st.markdown("""
    La calibration du modèle de Heston consiste à déterminer les paramètres du modèle qui permettent de reproduire au mieux les prix observés sur le marché des options.
    Ce modèle repose sur une dynamique à volatilité stochastique, où le prix de l’actif suit une loi log-normale, et sa variance est elle-même modélisée par un second processus aléatoire (un mouvement brownien) corrélé au premier.

    Cette approche permet de mieux capturer des phénomènes de marché comme le smile de volatilité. La calibration utilise une méthode d’optimisation basée sur des simulations Monte Carlo, qui génèrent des trajectoires possibles du prix de l’actif.
    Un grand nombre de simulations (Nmc) améliore la précision, mais allonge aussi significativement le temps de calcul.
    """)

    # Lecture du fichier local
    option_data_affich = pd.read_csv("options.csv", sep=";")
    option_data_affich['Maturity'] = (pd.to_datetime(option_data_affich['expiration']) -
                               pd.to_datetime(option_data_affich['price_date'])).dt.days / 365.0

    st.success(f"{len(option_data_affich)} options chargées depuis options.csv")
    st.dataframe(option_data_affich.head(7))



    # Afficher les paramètres par défaut
    st.markdown("## 🔍 Paramètres calibrés existants")
    df_default = pd.DataFrame({
        "Paramètre": ["v0", "rho", "theta", "k", "eta"],
        "Valeur": list(DEFAULT_PARAMS.values()),
        "Description": [
            "Variance initiale",
            "Corrélation actif/vol",
            "Variance long-terme",
            "Vitesse de retour à la moyenne",
            "Volatilité de la volatilité"
        ]
    })
    st.table(df_default.style.format({"Valeur": "{:.6f}"}))

    st.info("ℹ️ Ces paramètres ont été pré-calibrés (Nmc=20 000 >1H) et sont utilisés par défaut dans le pricer.")


    # Saisie des paramètres par l'utilisateur
    st.subheader("⚙️ Paramètres de calibration")
    col1, col2, col3 = st.columns(3)
    with col1:
        S0 = st.number_input("Prix Spot de votre action (S0)", value=215.0)
    with col2:
        r = st.number_input("Taux sans risque (r)", value=0.045)
    with col3:
        Nmc = st.slider("Nombre de simulations Monte Carlo (Nmc)",
                        min_value=100,
                        max_value=100000,
                        value=300,  # Valeur par défaut augmentée pour plus de stabilité
                        step=100)

    if st.button("📈 Lancer la calibration"):
        with st.spinner(f"🧮 Calibration en cours avec Nmc={Nmc}... cela peut prendre un moment"):
            start_time = time.time()

            # Appel direct de la fonction qui utilise les options déjà filtrées
            option_data = load_option_data()
            optimal_params = calibrate_heston_simplified(option_data, S0, r, Nmc, 100)

            end_time = time.time()
            calibration_time = end_time - start_time

            v0, rho, theta, k, eta = optimal_params

            st.success(f"✅ Calibration terminée en {calibration_time:.2f} secondes")
            st.balloons()

            # Sauvegarde des résultats
            st.session_state.calibrated = True
            st.session_state.params = {
                'v0': v0,
                'rho': rho,
                'theta': theta,
                'k': k,
                'eta': eta
            }
            st.session_state.calibration_time = calibration_time


            st.balloons()

            # Affichage des résultats
            st.subheader("📊 Paramètres calibrés")
            df_params = pd.DataFrame({
                "Paramètre": ["v0", "rho", "theta", "k", "eta"],
                "Valeur": [v0, rho, theta, k, eta],
                "Description": [
                    "Variance initiale",
                    "Corrélation actif/vol",
                    "Variance long-terme",
                    "Vitesse de retour à la moyenne",
                    "Volatilité de la volatilité"
                ]
            })
            st.table(df_params.style.format({"Valeur": "{:.4f}"}))

            # Debug: Vérification des paramètres
            st.write("Détails de calibration:")
            st.write(f"- Temps de calcul: {calibration_time:.2f}s")
            st.write(f"- Valeur Nmc effective: {Nmc}")



    # Bouton pour afficher le smile de volatilité
    if st.button(" Afficher le Smile de Volatilité"):
        st.subheader("Smile de volatilité calculé avec le modèle de Heston")

        def show_calibration():
            st.title("🔧 Calibration du Modèle Heston")

            st.markdown(
                "Cette section permet de calibrer le modèle de Heston sur des options Apple et de fournir les cinq paramètres du modèle, qui nous seront très utiles pour la suite.")
            st.markdown(" Voici le temps de calibration indicatif par nombre de simulation de Monte Carlo ")
            data = {
                "100 Simulation": ["1 minute"],
                "1 000 Simulations": ["7 minutes"],
                "10 000 Simulations": ["45 minutes"]
            }

            df = pd.DataFrame(data)

            st.subheader("📋 Informations utiles")
            st.table(df)
            # Initialisation des paramètres dans session_state
            if 'calibrated' not in st.session_state:
                st.session_state.calibrated = False
                st.session_state.params = {
                    'v0': 0.117478,
                    'rho': -0.353219,
                    'theta': 0.063540,
                    'k': 1.251697,
                    'eta': 0.898913
                }

            st.markdown("""
            La calibration du modèle de Heston consiste à déterminer les paramètres du modèle qui permettent de reproduire au mieux les prix observés sur le marché des options.
            Ce modèle repose sur une dynamique à volatilité stochastique, où le prix de l’actif suit une loi normale, et sa variance est elle-même modélisée par un second processus aléatoire (un mouvement brownien) corrélé au premier.

            Cette approche permet de mieux capturer des phénomènes de marché comme le smile de volatilité. La calibration utilise une méthode d’optimisation basée sur des simulations Monte Carlo, qui génèrent des trajectoires possibles du prix de l’actif.
            Un grand nombre de simulations (Nmc) améliore la précision, mais allonge aussi significativement le temps de calcul.
            """)

            # Lecture du fichier local
            option_data_affich = pd.read_csv("options.csv", sep=";")
            option_data_affich['Maturity'] = (pd.to_datetime(option_data_affich['expiration']) -
                                              pd.to_datetime(option_data_affich['price_date'])).dt.days / 365.0

            st.success(f"{len(option_data_affich)} options chargées depuis options.csv")
            st.dataframe(option_data_affich.head(7))

            # Saisie des paramètres par l'utilisateur
            st.subheader("⚙️ Paramètres de calibration")
            col1, col2, col3 = st.columns(3)
            with col1:
                S0 = st.number_input("Prix Spot de votre action (S0)", value=215.0)
            with col2:
                r = st.number_input("Taux sans risque (r)", value=0.045)
            with col3:
                Nmc = st.slider("Nombre de simulations Monte Carlo (Nmc)",
                                min_value=100,
                                max_value=100000,
                                value=300,  # Valeur par défaut augmentée pour plus de stabilité
                                step=100)

            if st.button("📈 Lancer la calibration"):
                with st.spinner(f"🧮 Calibration en cours avec Nmc={Nmc}... cela peut prendre un moment"):
                    start_time = time.time()

                    # Appel direct de la fonction qui utilise les options déjà filtrées
                    option_data = load_option_data()
                    optimal_params = calibrate_heston_simplified(option_data, S0, r, Nmc, 100)

                    end_time = time.time()
                    calibration_time = end_time - start_time

                    v0, rho, theta, k, eta = optimal_params

                    st.success(f"✅ Calibration terminée en {calibration_time:.2f} secondes")
                    st.balloons()

                    # Sauvegarde des résultats
                    st.session_state.calibrated = True
                    st.session_state.params = {
                        'v0': v0,
                        'rho': rho,
                        'theta': theta,
                        'k': k,
                        'eta': eta
                    }
                    st.session_state.calibration_time = calibration_time

                    st.balloons()

                    # Affichage des résultats
                    st.subheader("📊 Paramètres calibrés")
                    df_params = pd.DataFrame({
                        "Paramètre": ["v0", "rho", "theta", "k", "eta"],
                        "Valeur": [v0, rho, theta, k, eta],
                        "Description": [
                            "Variance initiale",
                            "Corrélation actif/vol",
                            "Variance long-terme",
                            "Vitesse de retour à la moyenne",
                            "Volatilité de la volatilité"
                        ]
                    })
                    st.table(df_params.style.format({"Valeur": "{:.4f}"}))

                    # Debug: Vérification des paramètres
                    st.write("Détails de calibration:")
                    st.write(f"- Temps de calcul: {calibration_time:.2f}s")
                    st.write(f"- Valeur Nmc effective: {Nmc}")

            # Bouton pour afficher le smile de volatilité
            if st.button(" Afficher le Smile de Volatilité"):
                st.subheader("Smile de volatilité calculé avec le modèle de Heston")

                # Strikes autour du spot
                strikes = np.linspace(S0 * 0.7, S0 * 1.3, 20)

                df_smile = trace_smile_vol_heston(
                    S0=S0,
                    r=r,
                    maturite=1.0,  # ou un input utilisateur si besoin
                    option_type="call",
                    params=[
                        st.session_state.params['v0'],
                        st.session_state.params['rho'],
                        st.session_state.params['theta'],
                        st.session_state.params['k'],
                        st.session_state.params['eta']
                    ],
                    strikes=strikes,
                    Nmc=Nmc  # Nmc choisi par l'utilisateur
                )

        # Strikes autour du spot
        strikes = np.linspace(S0 * 0.7, S0 * 1.3, 20)

        df_smile = trace_smile_vol_heston(
            S0=S0,
            r=r,
            maturite=1.0,  # ou un input utilisateur si besoin
            option_type="call",
            params=[
                st.session_state.params['v0'],
                st.session_state.params['rho'],
                st.session_state.params['theta'],
                st.session_state.params['k'],
                st.session_state.params['eta']
            ],
            strikes=strikes,
            Nmc=Nmc  # Nmc choisi par l'utilisateur
        )
        st.dataframe(df_smile.style.format({"Prix Heston": "{:.4f}", "Vol Implicite": "{:.4%}"}))
# ======================================
# NAVIGATION
# ======================================
pages = {
    "Accueil": show_home,
    "Calibration Heston": show_calibration
}

selected_page = st.sidebar.selectbox("Navigation", list(pages.keys()))
pages[selected_page]()