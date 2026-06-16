import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def generate_robustness_heatmap():
    # 1. Load Data
    try:
        df = pd.read_csv("results/standardized_study.csv")
    except FileNotFoundError:
        print("Registry not found.")
        return

    # Filter for Spectral solver and M=150
    df = df[(df['solver'] == 'Spectral') & (df['m_points'] == 150)]
    
    # 2. Pivot for Heatmap (X=N, Y=K)
    # We take the mean RMSE if multiple trials exist
    pivot_df = df.pivot_table(index='k_neighbors', columns='n_users', values='mean_rmse', aggfunc='mean')
    pivot_df = pivot_df.sort_index(ascending=False) # K=10 at top

    # 3. Plotting
    plt.figure(figsize=(12, 10))
    
    # Log scale normalization: Green (0.005m) to Red (1.0m)
    norm = colors.LogNorm(vmin=0.005, vmax=1.0)
    
    heatmap = plt.imshow(pivot_df, cmap='RdYlGn_r', norm=norm, aspect='auto')
    
    # Add annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.iloc[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.1 else "black"
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontweight='bold')

    # Formatting
    plt.colorbar(heatmap, label='Mean RMSE (m) [Log Scale]')
    plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    
    plt.title('Spectral Robustness Heatmap (N vs. K)\nFixed Master World (Clustered & Stratified)', fontsize=14)
    plt.xlabel('Number of Users (N)', fontsize=12)
    plt.ylabel('Neighbors per Node (K)', fontsize=12)
    
    # Highlight the breakdown line
    plt.tight_layout()
    plot_path = "results/robustness_heatmap.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Heatmap updated: {plot_path}")

if __name__ == "__main__":
    generate_robustness_heatmap()
