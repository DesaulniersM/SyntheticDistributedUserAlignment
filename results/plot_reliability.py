import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_reliability_plot():
    # 1. Load Data
    try:
        df = pd.read_csv("results/standardized_study.csv")
    except FileNotFoundError:
        print("Registry not found.")
        return

    # Success threshold: RMSE < 1.0m
    df['success'] = df['mean_rmse'] < 1.0
    
    # We'll plot Success Rate vs N for K=3, K=4, K=6
    k_targets = [3, 4, 6]
    
    plt.figure(figsize=(10, 6), dpi=300)
    colors_spectral = ['#2ca02c', '#1f77b4', '#9467bd'] # Greens/Blues/Purples
    colors_gtsam = ['#d62728', '#ff7f0e', '#8c564b']    # Reds/Oranges/Browns
    
    for i, k in enumerate(k_targets):
        # Filter for K
        df_k = df[df['k_neighbors'] == k]
        
        # Spectral Success Rate
        spec_k = df_k[df_k['solver'] == 'Spectral'].groupby('n_users')['success'].mean() * 100
        plt.plot(spec_k.index, spec_k.values, marker='o', linestyle='-', 
                 color=colors_spectral[i], label=f'Spectral (K={k})', linewidth=2)
        
        # GTSAM Success Rate
        gtsam_k = df_k[df_k['solver'] == 'GTSAM_Cold'].groupby('n_users')['success'].mean() * 100
        plt.plot(gtsam_k.index, gtsam_k.values, marker='x', linestyle='--', 
                 color=colors_gtsam[i], label=f'GTSAM (K={k})', alpha=0.8)

    plt.title('Reliability Frontier: Success Rate at Scale\n(Success = RMSE < 1.0m)', fontsize=14)
    plt.xlabel('Number of Users (N)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = "results/reliability_frontier.png"
    plt.savefig(plot_path)
    print(f"Reliability plot saved: {plot_path}")

if __name__ == "__main__":
    generate_reliability_plot()
