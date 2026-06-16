import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def generate_small_world_heatmaps():
    try:
        df = pd.read_csv("results/small_world_study.csv")
    except FileNotFoundError:
        print("Registry not found.")
        return

    solvers = ['Spectral', 'GTSAM_Cold']
    titles = {'Spectral': 'Spectral Robustness (Small World - 20m)', 
              'GTSAM_Cold': 'GTSAM Robustness (Small World - 20m)'}
    filenames = {'Spectral': 'results/small_world_spectral_heatmap.png',
                 'GTSAM_Cold': 'results/small_world_gtsam_heatmap.png'}

    for solver in solvers:
        df_s = df[(df['solver'] == solver) & (df['m_points'] == 150)]
        if df_s.empty: continue
        
        pivot_df = df_s.pivot_table(index='k_neighbors', columns='n_users', values='mean_rmse', aggfunc='mean')
        pivot_df = pivot_df.sort_index(ascending=False)

        plt.figure(figsize=(10, 8))
        norm = colors.LogNorm(vmin=0.005, vmax=1.0)
        heatmap = plt.imshow(pivot_df, cmap='RdYlGn_r', norm=norm, aspect='auto')
        
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                val = pivot_df.iloc[i, j]
                if not np.isnan(val):
                    color = "white" if val > 0.1 else "black"
                    plt.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontweight='bold')

        plt.colorbar(heatmap, label='Mean RMSE (m) [Log Scale]')
        plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
        plt.yticks(range(len(pivot_df.index)), pivot_df.index)
        plt.title(titles[solver])
        plt.xlabel('Number of Users (N)')
        plt.ylabel('Neighbors (K)')
        plt.tight_layout()
        plt.savefig(filenames[solver], dpi=300)
        plt.close()
        print(f"Saved: {filenames[solver]}")

if __name__ == "__main__":
    generate_small_world_heatmaps()
