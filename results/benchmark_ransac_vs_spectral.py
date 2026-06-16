import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stress_test_spectral import SpectralStressTester

def run_solver_comparison(n_trials: int = 1):
    m_points = 150 # Target Sweet Spot
    n_users = 60
    
    methods = ["RANSAC+SVD Local"] # Only run RANSAC
    results = {m: {"rmse": [], "time": []} for m in methods}

    print(f"\n>>> Comparing Solvers ({n_users} Users, M={m_points})")
    print("-" * 50)

    for method in methods:
        use_ransac = (method == "RANSAC+SVD Local")
        tester = SpectralStressTester(use_ransac=use_ransac)
        
        for t in range(n_trials):
            # Same parameters for both
            tester.generate_synthetic_data(n_users=n_users, topology='grid', n_landmarks=m_points, 
                                         outlier_ratio=0.1, noise_std=0.02, max_visibility_dist=6.0)
            
            res = tester.run_test(neighbors_per_node=3)
            results[method]["rmse"].append(res["mean_rmse"])
            results[method]["time"].append(res["duration"])
            
        avg_rmse = np.mean(results[method]["rmse"])
        avg_time = np.mean(results[method]["time"])
        print(f"{method:<20} | Mean RMSE: {avg_rmse:<10.4f} | Mean Time: {avg_time:<10.4f}")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Time Comparison
    ax1.bar(methods, [np.mean(results[m]["time"]) for m in methods], color=['blue', 'green'], alpha=0.7)
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Computational Latency (Lower is Better)')

    # RMSE Comparison
    ax2.bar(methods, [np.mean(results[m]["rmse"]) for m in methods], color=['red', 'orange'], alpha=0.7)
    ax2.set_ylabel('Mean RMSE (m)')
    ax2.set_title('Global Accuracy (Lower is Better)')
    ax2.set_yscale('log') # Log scale helps see small differences

    plt.suptitle(f'Head-to-Head: Local Solver Comparison ($N={n_users}, M={m_points}$)')
    plt.tight_layout()
    
    plot_path = "results/solver_comparison.png"
    plt.savefig(plot_path)
    print(f"\nComparison graph saved to: {plot_path}")

if __name__ == "__main__":
    run_solver_comparison()
