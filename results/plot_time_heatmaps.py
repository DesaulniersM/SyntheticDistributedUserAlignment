import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

def generate_time_heatmaps():
    # 1. Load Data
    try:
        df = pd.read_csv("results/standardized_study.csv")
    except FileNotFoundError:
        print("Registry not found.")
        return

    solvers = ['Spectral', 'GTSAM_Cold']
    filenames = {
        'Spectral': 'results/spectral_time_heatmap.png',
        'GTSAM_Cold': 'results/gtsam_time_heatmap.png'
    }

    # Find global max duration for consistent scale
    max_duration = df['duration'].quantile(0.95) # Use 95th percentile to avoid outlier distortion

    for solver in solvers:
        # Filter for the solver and standard M=150
        df_solver = df[(df['solver'] == solver) & (df['m_points'] == 150)]
        
        if df_solver.empty:
            print(f"No data found for {solver}")
            continue

        # Pivot to get the mean of 'duration'
        pivot_df = df_solver.pivot_table(index='k_neighbors', columns='n_users', values='duration', aggfunc='mean')
        pivot_df = pivot_df.sort_index(ascending=False) 

        # 3. Plotting
        plt.figure(figsize=(12, 10))
        
        cmap = 'plasma'
        norm = colors.Normalize(vmin=0.0, vmax=max_duration)
        
        heatmap = plt.imshow(pivot_df, cmap=cmap, norm=norm, aspect='auto')
        
        # Add annotations
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                val = pivot_df.iloc[i, j]
                if not np.isnan(val):
                    text_val = f"{val:.1f}s"
                    color = "white" if val > (max_duration * 0.6) else "black"
                    plt.text(j, i, text_val, ha="center", va="center", color=color, fontweight='bold', fontsize=8)

        # Formatting
        plt.colorbar(heatmap, label='Average Solve Time (seconds)')
        plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
        plt.yticks(range(len(pivot_df.index)), pivot_df.index)
        
        plt.title(f'{solver} Average Solve Time (M=150)', fontsize=14)
        plt.xlabel('Number of Users (N)', fontsize=12)
        plt.ylabel('Neighbors per Node (K)', fontsize=12)
        
        plt.tight_layout()
        plot_path = filenames[solver]
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Time Heatmap saved: {plot_path}")

if __name__ == "__main__":
    generate_time_heatmaps()
