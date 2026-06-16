import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

def generate_probability_heatmaps(threshold=0.05):
    # 1. Load Data
    try:
        df = pd.read_csv("results/standardized_study.csv")
    except FileNotFoundError:
        print("Registry not found.")
        return

    # Define success based on the new threshold
    df['success'] = df['mean_rmse'] < threshold

    solvers = ['Spectral', 'GTSAM_Cold']
    titles = {
        'Spectral': f'Spectral Probability of High Precision (RMSE < {threshold}m)',
        'GTSAM_Cold': f'GTSAM Probability of High Precision (RMSE < {threshold}m)'
    }
    filenames = {
        'Spectral': 'results/spectral_prob_heatmap9cm.png',
        'GTSAM_Cold': 'results/gtsam_prob_heatmap9cm.png'
    }

    for solver in solvers:
        # Filter for the solver and standard M=150
        df_solver = df[(df['solver'] == solver) & (df['m_points'] == 150)]
        
        if df_solver.empty:
            print(f"No data found for {solver}")
            continue

        # Pivot to get the mean of 'success' (which equates to probability 0.0 to 1.0)
        pivot_df = df_solver.pivot_table(index='k_neighbors', columns='n_users', values='success', aggfunc='mean')
        pivot_df = pivot_df.sort_index(ascending=False) 

        # 3. Plotting
        plt.figure(figsize=(12, 10))
        
        # Use a colormap from red (0) to green (1)
        cmap = 'RdYlGn'
        norm = colors.Normalize(vmin=0.0, vmax=1.0)
        
        heatmap = plt.imshow(pivot_df, cmap=cmap, norm=norm, aspect='auto')
        
        # Add annotations
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                val = pivot_df.iloc[i, j]
                if not np.isnan(val):
                    # Display as percentage
                    text_val = f"{val*100:.0f}%"
                    color = "white" if (val < 0.3 or val > 0.7) else "black"
                    plt.text(j, i, text_val, ha="center", va="center", color=color, fontweight='bold', fontsize=9)

        # Formatting
        plt.colorbar(heatmap, label=f'Probability of RMSE < {threshold}m')
        plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
        plt.yticks(range(len(pivot_df.index)), pivot_df.index)
        
        plt.title(titles[solver], fontsize=14)
        plt.xlabel('Number of Users (N)', fontsize=12)
        plt.ylabel('Neighbors per Node (K)', fontsize=12)
        
        plt.tight_layout()
        plot_path = filenames[solver]
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Probability Heatmap saved: {plot_path}")

if __name__ == "__main__":
    generate_probability_heatmaps(threshold=0.1)
