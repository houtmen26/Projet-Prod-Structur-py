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
        self.params_heston = [0.04, -0.7, 0.04, 1.0, 0.2]  # v0, rho, theta, k, eta
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
        ui_call = OptionBarriereUpAndInCall(...).price()[0]  # (mêmes paramètres que précédemment)
        uo_call = OptionBarriereUpAndOutCall(...).price()[0]
        call_vanilla = Call(...).price()

        ecart = abs((ui_call + uo_call) - call_vanilla)
        print(f"Call UI + UO vs Vanilla: écart = {ecart:.4f}€ | {'OK' if ecart < 0.5 else 'ATTENTION'}")

        # 2. Put-Call Parity pour options barrières
        # ... (implémentation similaire)


# ======================
# LANCEMENT DES TESTS
# ======================

if __name__ == "__main__":
    tester = TestOptionsBarrieres()
    tester.run_tests()