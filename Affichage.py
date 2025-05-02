import streamlit as st
import pandas as pd
import numpy as np
import time
from scipy.optimize import minimize
from Modele_Heston import *
from Call_Heston import *
from produit_financier import *  # Module avec les classes de produits structurés

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
        "1 000 Simulations": ["7 minutes"],
        "10 000 Simulations": ["45 minutes"]
    }

    df = pd.DataFrame(data)

    st.table(df)

    # Initialisation des paramètres dans session_state
    if 'params' not in st.session_state:
        st.session_state.params = DEFAULT_PARAMS
        st.session_state.calibrated = False  # False car ce sont des valeurs par défaut

    st.markdown("""
    La calibration du modèle de Heston consiste à déterminer les paramètres du modèle qui permettent de reproduire au mieux les prix observés sur le marché des options.
    Ce modèle repose sur une dynamique à volatilité stochastique, où le prix de l'actif suit une loi log-normale, et sa variance est elle-même modélisée par un second processus aléatoire (un mouvement brownien) corrélé au premier.

    Cette approche permet de mieux capturer des phénomènes de marché comme le smile de volatilité. La calibration utilise une méthode d'optimisation basée sur des simulations Monte Carlo, qui génèrent des trajectoires possibles du prix de l'actif.
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
                        value=300,
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
            maturite=1.0,
            option_type="call",
            params=[
                st.session_state.params['v0'],
                st.session_state.params['rho'],
                st.session_state.params['theta'],
                st.session_state.params['k'],
                st.session_state.params['eta']
            ],
            strikes=strikes,
            Nmc=Nmc
        )
        st.dataframe(df_smile.style.format({"Prix Heston": "{:.4f}", "Vol Implicite": "{:.4%}"}))


# ======================================
# SECTION 3 : PRICER DE PRODUITS STRUCTURÉS
# ======================================
def show_pricer():
    st.title("💰 Pricer de Produits Structurés")
    st.markdown("""
    Cette section permet de calculer le prix de différents produits structurés en utilisant le modèle de Heston.
    Les paramètres par défaut sont ceux pré-calibrés (Nmc=20 000), mais vous pouvez les modifier manuellement.
    """)

    # Sélection du produit
    produit = st.selectbox("Choisissez un produit :", [
        "Call", "Put", "CallSpread", "Straddle", "Strangle", "Strip", "Strap",
        "OptionCallBinaire", "ReverseConvertible", "Autocall"])

    # Paramètres communs
    st.subheader("Paramètres généraux")
    col1, col2, col3 = st.columns(3)
    with col1:
        S0 = st.number_input("Prix spot (S0)", value=100.0)
    with col2:
        maturite = st.number_input("Maturité (en années)", value=1.0)
    with col3:
        r = st.number_input("Taux sans risque (r)", value=0.03)

    # Paramètres de Heston (pré-remplis avec les valeurs calibrées ou par défaut)
    st.subheader("Paramètres du modèle de Heston")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        v0 = st.number_input("v0", value=st.session_state.params.get('v0', DEFAULT_PARAMS['v0']))
    with col2:
        rho = st.number_input("rho", value=st.session_state.params.get('rho', DEFAULT_PARAMS['rho']))
    with col3:
        theta = st.number_input("theta", value=st.session_state.params.get('theta', DEFAULT_PARAMS['theta']))
    with col4:
        k = st.number_input("k", value=st.session_state.params.get('k', DEFAULT_PARAMS['k']))
    with col5:
        eta = st.number_input("eta", value=st.session_state.params.get('eta', DEFAULT_PARAMS['eta']))
    param = [v0, rho, theta, k, eta]

    # Configuration Monte Carlo
    st.subheader("Paramètres de simulation")
    col1, col2 = st.columns(2)
    with col1:
        Nmc = st.slider("Nombre de simulations Monte Carlo", 1000, 100000, 10000, step=1000)
    with col2:
        N = st.slider("Nombre de pas", 50, 500, 200, step=10)

    mc_config = MonteCarloConfig(Nmc=Nmc, N=N, seed=42)
    action = Action("Asset", S0)

    # Configuration spécifique au produit
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
        with st.spinner("Calcul en cours... (Cette opération peut prendre plusieurs minutes)"):
            try:
                prix = instrument.price()
                st.success(f"💰 Prix estimé : {prix:.4f} €")
                st.balloons()

                # Affichage des paramètres utilisés
                st.markdown("### Paramètres utilisés pour le calcul :")
                params_df = pd.DataFrame({
                    "Paramètre": ["S0", "Maturité", "Taux sans risque", "v0", "rho", "theta", "k", "eta", "Nmc", "N"],
                    "Valeur": [S0, maturite, r, v0, rho, theta, k, eta, Nmc, N]
                })
                st.table(params_df.style.format({"Valeur": "{:.4f}"}))

            except Exception as e:
                st.error(f"Une erreur est survenue lors du calcul : {str(e)}")
                st.error("Veuillez vérifier vos paramètres et réessayer.")


# ======================================
# NAVIGATION
# ======================================
pages = {
    "Accueil": show_home,
    "Calibration Heston": show_calibration,
    "Pricer Structuré": show_pricer
}

selected_page = st.sidebar.selectbox("Navigation", list(pages.keys()))
pages[selected_page]()