import numpy as np
from typing import Union
from produit_financier import *


# ======================
# CLASSES DE TEST
# ======================

class TestOptionsBarrieres:
    def __init__(self):
        # Paramètres communs
        self.S0 = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.params_heston = [0.117478, -0.353219, 0.063540, 1.251697, 0.898913 ]
        self.mc_config = MonteCarloConfig(Nmc=100000, N=252, seed=42)
        self.action = Action("TEST", self.S0)

        # Barrières pour les tests
        self.barriere_high = 120.0  # +20%
        self.barriere_low = 80.0  # -20%

    def run_tests(self):
        print("=== TEST DE COHÉRENCE DES OPTIONS BARRIÈRES ===")
        print(f"Spot: {self.S0} | Strike: {self.K} | Maturité: {self.T} an(s)")
        print(f"Paramètres Heston: {self.params_heston}")
        print("\n")

        # 1. Test des calls barrières
        self._test_call_barrieres()

        # 2. Test des puts barrières
        self._test_put_barrieres()

        # 3. Test de la note capital protégé
        self._test_note_capital_protege()

        # 4. Vérifications d'arbitrage
        self._verify_arbitrage()

    def _test_call_barrieres(self):
        print("=== CALLS BARRIÈRES ===")

        # Up-and-In
        ui_call = OptionBarriereUpAndIn_CALL(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            barriere=self.barriere_high,
            mc_config=self.mc_config
        )
        prix_ui, prob_ui = ui_call.price()
        print(f"Up-and-In Call (H={self.barriere_high}): {prix_ui:.2f}€ | Prob activation: {prob_ui:.1%}")

        # Up-and-Out
        uo_call = OptionBarriereUpAndOutCall(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            barriere=self.barriere_high,
            mc_config=self.mc_config
        )
        prix_uo, prob_uo = uo_call.price()
        print(f"Up-and-Out Call (H={self.barriere_high}): {prix_uo:.2f}€ | Prob désactivation: {prob_uo:.1%}")

        # Down-and-In
        di_call = OptionBarriereDownAndInCall(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            barriere=self.barriere_low,
            mc_config=self.mc_config
        )
        prix_di, prob_di = di_call.price()
        print(f"Down-and-In Call (H={self.barriere_low}): {prix_di:.2f}€ | Prob activation: {prob_di:.1%}")

        # Down-and-Out
        do_call = OptionBarriereDownAndOutCall(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            barriere=self.barriere_low,
            mc_config=self.mc_config
        )
        prix_do, prob_do = do_call.price()
        print(f"Down-and-Out Call (H={self.barriere_low}): {prix_do:.2f}€ | Prob désactivation: {prob_do:.1%}")

        # Call vanilla pour référence
        call_vanilla = Call(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            mc_config=self.mc_config
        ).price()
        print(f"\nCall Vanilla: {call_vanilla:.2f}€")

    def _test_put_barrieres(self):
        print("\n=== PUTS BARRIÈRES ===")

        # Up-and-In
        ui_put = OptionBarriereUpAndInPut(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            barriere=self.barriere_high,
            mc_config=self.mc_config
        )
        prix_ui, prob_ui = ui_put.price()
        print(f"Up-and-In Put (H={self.barriere_high}): {prix_ui:.2f}€ | Prob activation: {prob_ui:.1%}")

        # Up-and-Out
        uo_put = OptionBarriereUpAndOutPut(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            barriere=self.barriere_high,
            mc_config=self.mc_config
        )
        prix_uo, prob_uo = uo_put.price()
        print(f"Up-and-Out Put (H={self.barriere_high}): {prix_uo:.2f}€ | Prob désactivation: {prob_uo:.1%}")

        # Put vanilla pour référence
        put_vanilla = Put(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            mc_config=self.mc_config
        ).price()
        print(f"\nPut Vanilla: {put_vanilla:.2f}€")

    def _test_note_capital_protege(self):
        print("\n=== NOTE À CAPITAL PROTÉGÉ ===")

        note = NoteCapitalProtegee(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            barriere=self.barriere_high,
            rebate=50,  # 50€ si barrière touchée
            nominal=1000,
            mc_config=self.mc_config
        )
        prix_note = note.price()
        print(f"Prix de la note (nominal=1000€): {prix_note:.2f}€")
        print(f"Rendement implicite: {(prix_note / 1000 - 1):.2%}")


    def _verify_arbitrage(self):
        print("\n=== VÉRIFICATIONS D'ARBITRAGE ===")

        # 1. Call Up-and-In + Up-and-Out = Call Vanilla
        ui_call = OptionBarriereUpAndIn_CALL(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            barriere=self.barriere_high,
            mc_config=self.mc_config
        ).price()[0]

        uo_call = OptionBarriereUpAndOutCall(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            barriere=self.barriere_high,
            mc_config=self.mc_config
        ).price()[0]

        call_vanilla = Call(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            mc_config=self.mc_config
        ).price()

        ecart = abs((ui_call + uo_call) - call_vanilla)
        print(f"Call UI + UO vs Vanilla: écart = {ecart:.4f}€ | {'OK' if ecart < 0.5 else 'ATTENTION'}")

        # 2. Put-Call Parity pour options barrières
        # Down-and-In Call + Down-and-Out Call = Call Vanilla
        di_call = OptionBarriereDownAndInCall(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            barriere=self.barriere_low,
            mc_config=self.mc_config
        ).price()[0]

        do_call = OptionBarriereDownAndOutCall(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            barriere=self.barriere_low,
            mc_config=self.mc_config
        ).price()[0]

        ecart2 = abs((di_call + do_call) - call_vanilla)
        print(f"Call DI + DO vs Vanilla: écart = {ecart2:.4f}€ | {'OK' if ecart2 < 0.5 else 'ATTENTION'}")

        # 3. Put-Call Parity standard
        put_vanilla = Put(
            sous_jacent=self.action,
            maturite=self.T,
            parametres=self.params_heston,
            r=self.r,
            strike=self.K,
            mc_config=self.mc_config
        ).price()

        S0 = self.action.S0
        K = self.K
        T = self.T
        r = self.r

        ecart_pcp = abs(call_vanilla - put_vanilla - S0 + K * np.exp(-r * T))
        print(f"\nPut-Call Parity standard:")
        print(f"Call - Put = {call_vanilla:.2f} - {put_vanilla:.2f} = {call_vanilla - put_vanilla:.2f}")
        print(f"S0 - K*exp(-rT) = {S0:.2f} - {K * np.exp(-r * T):.2f} = {S0 - K * np.exp(-r * T):.2f}")
        print(f"Écart: {ecart_pcp:.4f}€ | {'OK' if ecart_pcp < 0.5 else 'ATTENTION'}")


# ======================
# LANCEMENT DES TESTS
# ======================

if __name__ == "__main__":
    tester = TestOptionsBarrieres()
    tester.run_tests()
    # ======= PARAMÈTRES COMMUNS =======
    # Sous-jacent fictif
    action_test = Action("TEST", 100)

    # Paramètres Heston pour la simulation
    param_heston = [0.04, -0.7, 0.04, 1.0, 0.2]  # v0, rho, theta, k, eta
    r = 0.05  # Taux sans risque
    maturite = 1.0  # 1 an
    mc_config = MonteCarloConfig(Nmc=10000, N=252, seed=42)

    # ======= TEST DE CHAQUE PRODUIT =======

    # Test CallSpread
    callspread = CallSpread(
        sous_jacent=action_test,
        maturite=maturite,
        parametres=param_heston,
        r=r,
        strikes=[90, 110],  # Strike bas et strike haut
        mc_config=mc_config
    )
    print(f"Prix du Call Spread : {callspread.price():.2f} €")
    callspread.plot_payoff()

    # Test Straddle
    straddle = Straddle(
        sous_jacent=action_test,
        maturite=maturite,
        parametres=param_heston,
        r=r,
        strikes=[100],  # Strike commun
        mc_config=mc_config
    )
    print(f"Prix du Straddle : {straddle.price():.2f} €")
    straddle.plot_payoff()

    # Test Strangle
    strangle = Strangle(
        sous_jacent=action_test,
        maturite=maturite,
        parametres=param_heston,
        r=r,
        strikes=[95, 105],  # Strike Put et Call
        mc_config=mc_config
    )
    print(f"Prix du Strangle : {strangle.price():.2f} €")
    strangle.plot_payoff()

    # Test Strip
    strip = Strip(
        sous_jacent=action_test,
        maturite=maturite,
        parametres=param_heston,
        r=r,
        strikes=[100],  # Strike unique
        mc_config=mc_config
    )
    print(f"Prix du Strip : {strip.price():.2f} €")
    strip.plot_payoff()

    # Test Strap
    strap = Strap(
        sous_jacent=action_test,
        maturite=maturite,
        parametres=param_heston,
        r=r,
        strikes=[100],  # Strike unique
        mc_config=mc_config
    )
    print(f"Prix du Strap : {strap.price():.2f} €")
    strap.plot_payoff()
