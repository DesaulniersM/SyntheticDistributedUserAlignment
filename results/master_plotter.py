import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

def main():
    # 1. Load Data
    registry_path = "results/experiment_registry.csv"
    if not os.path.exists(registry_path):
        print(f"Error: {registry_path} not found.")
        return

    df = pd.read_csv(registry_path)
    
    # 2. Filter for m_points == 150 and specific solvers
    df = df[df['m_points'] == 150]
    df_spectral = df[df['solver'] == 'Spectral']
    df_gtsam = df[df['solver'] == 'GTSAM_Cold']

    if df_spectral.empty or df_gtsam.empty:
        print("Warning: One or both solvers (Spectral, GTSAM_Cold) have no data for M=150.")
        if df_spectral.empty: print("Missing: Spectral")
        if df_gtsam.empty: print("Missing: GTSAM_Cold")

    # 3. Aggregate Mean RMSE for each (N, K) cell
    pivot_spectral = df_spectral.pivot_table(index='k_neighbors', columns='n_users', values='mean_rmse', aggfunc='mean')
    pivot_gtsam = df_gtsam.pivot_table(index='k_neighbors', columns='n_users', values='mean_rmse', aggfunc='mean')

    # Sort indices so K increases upwards or downwards as per existing style (plot_heatmap used sort_index(ascending=False))
    pivot_spectral = pivot_spectral.sort_index(ascending=False)
    pivot_gtsam = pivot_gtsam.sort_index(ascending=False)

    # --- Visualization 1: GTSAM Robustness Heatmap ---
    plot_gtsam_heatmap(pivot_gtsam)

    # --- Visualization 2: Spectral vs GTSAM Delta Map ---
    plot_delta_map(pivot_spectral, pivot_gtsam)

def plot_gtsam_heatmap(pivot_df):
    plt.figure(figsize=(12, 10))
    
    # Log scale normalization: Green (0.005m) to Red (1.0m)
    norm = colors.LogNorm(vmin=0.005, vmax=1.0)
    
    heatmap = plt.imshow(pivot_df, cmap='RdYlGn_r', norm=norm, aspect='auto')
    
    # Add annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.iloc[i, j]
            if not np.isnan(val):
                # Robustness: Handle GTSAM failed (RMSE=10.0m)
                # If val > 1.0, it will be clamped by LogNorm or look very red
                color = "white" if val > 0.1 else "black"
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontweight='bold')

    plt.colorbar(heatmap, label='Mean RMSE (m) [Log Scale]')
    plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    
    plt.title('GTSAM_Cold Robustness Heatmap (N vs. K)\nFixed Master World (M=150)', fontsize=14)
    plt.xlabel('Number of Users (N)', fontsize=12)
    plt.ylabel('Neighbors per Node (K)', fontsize=12)
    
    plt.tight_layout()
    save_path = "results/gtsam_robustness_heatmap.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_delta_map(pivot_spectral, pivot_gtsam):
    # Ensure both have the same indices and columns
    # Find common N and K
    common_n = sorted(list(set(pivot_spectral.columns) & set(pivot_gtsam.columns)))
    common_k = sorted(list(set(pivot_spectral.index) & set(pivot_gtsam.index)), reverse=True)

    if not common_n or not common_k:
        print("Error: No common (N, K) cells between Spectral and GTSAM.")
        return

    ps = pivot_spectral.loc[common_k, common_n]
    pg = pivot_gtsam.loc[common_k, common_n]

    # RPI = log10(Spectral_RMSE / GTSAM_RMSE)
    # Positive: Spectral is worse (GTSAM better) -> Pink/Warm
    # Negative: Spectral is better -> Green/Cool
    rpi = np.log10(ps / pg)

    # % Accuracy Difference: ((GTSAM - Spectral) / GTSAM) * 100
    # Positive: Spectral is more accurate -> Spectral is better
    diff_pct = ((pg - ps) / pg) * 100

    plt.figure(figsize=(14, 11))
    
    # Colormap: 'PiYG' (Spectral Pink/Green)
    # Spectral is Green (better), GTSAM is Pink (better)
    # Actually RPI = log10(S/G). 
    # If S=0.01, G=0.1, RPI = log10(0.1) = -1 (Spectral better)
    # If S=0.1, G=0.01, RPI = log10(10) = 1 (GTSAM better)
    # In PiYG: Green is positive, Pink is negative? No, usually Green is positive.
    # Let's check: plt.cm.PiYG(0.0) is pinkish, 1.0 is greenish.
    # So if RPI is negative (Spectral better), we want it greenish? 
    # Wait, RPI = log10(S/G). If S < G, RPI < 0.
    # If we want Spectral better to be Green, and RPI < 0 for Spectral better, 
    # then we want negative to be Green. 
    # PiYG_r has Green at negative and Pink at positive.
    
    # Actually, RPI = log10(Spectral / GTSAM)
    # S=0.01, G=0.1 => RPI = -1. (Spectral better) -> Green
    # S=0.1, G=0.01 => RPI = 1. (GTSAM better) -> Pink
    
    cmap = plt.cm.PiYG_r # Green for negative, Pink for positive
    
    # vmin/vmax should be symmetric for "White = Tie" at 0
    # Range of 2 orders of magnitude?
    divnorm = colors.TwoSlopeNorm(vmin=-2.0, vcenter=0, vmax=2.0)
    
    heatmap = plt.imshow(rpi, cmap=cmap, norm=divnorm, aspect='auto')
    
    # Annotate with % Accuracy Difference
    for i in range(len(common_k)):
        for j in range(len(common_n)):
            val_rpi = rpi.iloc[i, j]
            val_pct = diff_pct.iloc[i, j]
            if not np.isnan(val_rpi):
                color = "black"
                # If RPI is very large (one is much better), text might need to be white
                if abs(val_rpi) > 1.2:
                    color = "white"
                
                # Handling cases where GTSAM failed (RMSE=10.0m)
                # If pg=10.0 and ps=0.01, diff_pct = (10-0.01)/10 * 100 = 99.9%
                # If pg=0.01 and ps=10.0, diff_pct = (0.01-10)/0.01 * 100 = -99900%
                
                if val_pct > 0:
                    label = f"+{val_pct:.1f}%"
                else:
                    label = f"{val_pct:.1f}%"
                
                plt.text(j, i, label, ha="center", va="center", color=color, fontweight='bold', fontsize=9)

    cbar = plt.colorbar(heatmap, label='Relative Performance Index: log10(Spectral / GTSAM)')
    plt.xticks(range(len(common_n)), common_n)
    plt.yticks(range(len(common_k)), common_k)
    
    plt.title("Relative Performance Map: Spectral vs. GTSAM (N vs. K)\nGreen: Spectral Better | Pink: GTSAM Better | % = (GTSAM - Spectral)/GTSAM", fontsize=14)
    plt.xlabel('Number of Users (N)', fontsize=12)
    plt.ylabel('Neighbors per Node (K)', fontsize=12)
    
    plt.tight_layout()
    save_path = "results/spectral_vs_gtsam_delta_map.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
