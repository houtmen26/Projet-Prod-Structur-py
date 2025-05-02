from Call_Heston import heston_option_price_put, heston_option_price_call
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from produit_financier import *
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns


def stress_test(product, spot_shocks=None, vol_shocks=None, rate_shocks=None):
    original_S0 = product.sous_jacent.S0
    original_params = product.parametres.copy()
    original_r = product.r

    spot_shocks = spot_shocks or [0.8, 0.9, 1.0, 1.1, 1.2]
    vol_shocks = vol_shocks or [0.5, 1.0, 1.5]
    rate_shocks = rate_shocks or [-0.01, 0.0, 0.01, 0.02]

    results = []

    for s_mult in spot_shocks:
        for v_mult in vol_shocks:
            for dr in rate_shocks:
                product.sous_jacent.S0 = original_S0 * s_mult
                product.parametres[0] = original_params[0] * v_mult
                product.r = original_r + dr

                price = product.price()
                price_val = price[0] if isinstance(price, tuple) else price
                results.append({
                    'S0': product.sous_jacent.S0,
                    'v0': product.parametres[0],
                    'r': product.r,
                    'Prix': price_val
                })

    # Reset original values
    product.sous_jacent.S0 = original_S0
    product.parametres = original_params
    product.r = original_r

    return pd.DataFrame(results)


def plot_stress_surface(df, original_spot):
    plt.figure(figsize=(10, 8))

    # Create pivot table for heatmap
    pivot = df[df['r'] == df['r'].unique()[1]].pivot(index='S0', columns='v0', values='Prix')

    # Create heatmap
    ax = sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu",
                     cbar_kws={'label': 'Option Price'})

    # Add reference line for original spot
    spot_values = pivot.index.values
    closest_idx = np.abs(spot_values - original_spot).argmin()
    ax.axhline(len(pivot) - closest_idx - 0.5, color='red', linestyle='--', linewidth=1)

    plt.title(f"Stress Test: Sensi du prix au Spot et à la vol \n(At r={df['r'].unique()[1]:.2f})")
    plt.xlabel("Volatilité (v0)")
    plt.ylabel("Prix Spot (S0)")
    plt.tight_layout()
    plt.show()


# Configuration
params = [0.117478, -0.353219, 0.063540, 1.251697, 0.898913]
action = Action("AAPL", 215)
mc_config = MonteCarloConfig(Nmc=10000, N=252, seed=42)

# Create product
call_spread_test = CallSpread(
    sous_jacent=action,
    maturite=1,
    parametres=params,
    r=0.05,
    strikes=[230,260],
    mc_config=mc_config
)

def export_stress_to_csv(df, filename="stress_test_results.csv"):
    """Exporte les résultats du stress test en CSV."""
    df.to_csv(filename, index=False)
    print(f"Fichier exporté : {filename}")

# Run stress test
df_stress = stress_test(call_spread_test)
print(df_stress.head())

# Plot results
plot_stress_surface(df_stress, action.S0)

# Additional visualization - Rate impact
plt.figure(figsize=(10, 6))
for v0 in df_stress['v0'].unique():
    subset = df_stress[df_stress['v0'] == v0]
    plt.plot(subset['r'], subset['Prix'], 'o-', label=f'v0={v0:.3f}')

plt.axvline(0.05, color='gray', linestyle='--', label='Original r')
plt.xlabel("Interest Rate (r)")
plt.ylabel("Option Price")
plt.title("Price Sensitivity to Interest Rates")
plt.legend()
plt.grid(True)
plt.show()