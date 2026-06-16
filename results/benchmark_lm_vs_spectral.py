import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stress_test_spectral import SpectralStressTester

def run_density_sweep(n_trials: int = 1):
    m_points = 150 
    n_users = 60
    # Sweep from highly sparse (2) to very dense (15)
    neighbor_counts = [2, 4, 6, 10, 15]
    
    results = {
        "k": neighbor_counts,
        "spectral_rmse": [],
        "lm_warm_rmse": [],
        "lm_cold_rmse": []
    }

    print(f"\n>>> Density Sweep: Accuracy vs. Connectivity (N={n_users})")
    print(f"{'K_Neighbors':<12} | {'Spectral RMSE':<15} | {'LM Cold RMSE':<15}")
    print("-" * 50)

    for k in neighbor_counts:
        s_rmses, lmc_rmses = [], []
        
        for t in range(n_trials):
            # 1. Generate Data (use any manager for this)
            gen_tester = SpectralStressTester(use_lm=False)
            gen_tester.generate_synthetic_data(n_users=n_users, topology='grid', n_landmarks=m_points, 
                                             outlier_ratio=0.1, noise_std=0.02, max_visibility_dist=10.0)
            
            # 2. Test Spectral
            s_tester = SpectralStressTester(use_lm=False)
            s_tester.manager.user_ids = list(gen_tester.manager.user_ids)
            s_tester.manager.user_clouds = gen_tester.manager.user_clouds.copy()
            s_tester.manager.user_features = gen_tester.manager.user_features.copy()
            s_tester.manager.user_gravities = gen_tester.manager.user_gravities.copy()
            s_tester.gt_poses = gen_tester.gt_poses
            
            res_s = s_tester.run_test(neighbors_per_node=k)
            s_rmses.append(res_s["mean_rmse"])
            
            # 3. Test LM Cold
            lm_tester = SpectralStressTester(use_lm=True)
            lm_tester.manager.user_ids = list(gen_tester.manager.user_ids)
            lm_tester.manager.user_clouds = gen_tester.manager.user_clouds.copy()
            lm_tester.manager.user_features = gen_tester.manager.user_features.copy()
            lm_tester.manager.user_gravities = gen_tester.manager.user_gravities.copy()
            lm_tester.gt_poses = gen_tester.gt_poses

            res_lmc = lm_tester.run_test(neighbors_per_node=k, lm_init='cold')
            lmc_rmses.append(res_lmc["mean_rmse"])
            
        avg_s = np.mean(s_rmses)
        avg_lmc = np.mean(lmc_rmses)
        results["spectral_rmse"].append(avg_s)
        results["lm_cold_rmse"].append(avg_lmc)
        
        print(f"{k:<12} | {avg_s:<15.4f} | {avg_lmc:<15.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(neighbor_counts, results["spectral_rmse"], 'o-', label='Spectral Sync (Initialization-Free)', linewidth=2)
    plt.plot(neighbor_counts, results["lm_cold_rmse"], 's--', label='LM (Cold Start)', linewidth=2)
    
    plt.yscale('log')
    plt.xlabel('Graph Connectivity (Neighbors per Node)')
    plt.ylabel('Mean RMSE (m) [Log Scale]')
    plt.title(f'Solver Robustness vs. Graph Density ($N={n_users}, M={150}$)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plot_path = "results/density_sweep.png"
    plt.savefig(plot_path)
    print(f"\nDensity sweep graph saved to: {plot_path}")

if __name__ == "__main__":
    run_density_sweep()

if __name__ == "__main__":
    run_global_solver_comparison()
