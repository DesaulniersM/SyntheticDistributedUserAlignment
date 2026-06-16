import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stress_test_spectral import SpectralStressTester

def run_high_res_salience_benchmark(n_trials: int = 5):
    tester = SpectralStressTester()
    
    # High-resolution sweep focusing on the transition zone
    landmark_counts = [50, 54, 58, 62, 66, 70, 75, 100, 150, 200, 300, 400, 500]
    n_users = 60
    
    results = {
        "m": [],
        "rmse_mean": [],
        "rmse_std": [],
        "time_mean": [],
        "edges_mean": []
    }

    print(f"\n>>> Starting Multi-Trial Salience Benchmark ({n_users} Users, {n_trials} Trials/Point)")
    print(f"{'M_Pts':<7} | {'Mean RMSE':<12} | {'Std Dev':<10} | {'Mean Time':<10} | {'Edges':<7}")
    print("-" * 65)

    for m in landmark_counts:
        trial_rmses = []
        trial_times = []
        trial_edges = []
        
        for t in range(n_trials):
            # Regenerate with new random seed each trial
            tester.generate_synthetic_data(n_users=n_users, topology='grid', n_landmarks=m, 
                                         outlier_ratio=0.1, noise_std=0.02, max_visibility_dist=6.0)
            
            res = tester.run_test(neighbors_per_node=3)
            trial_rmses.append(res["mean_rmse"])
            trial_times.append(res["duration"])
            trial_edges.append(res["edges_count"])
        
        m_rmse = np.mean(trial_rmses)
        s_rmse = np.std(trial_rmses)
        m_time = np.mean(trial_times)
        m_edges = np.mean(trial_edges)
        
        results["m"].append(m)
        results["rmse_mean"].append(m_rmse)
        results["rmse_std"].append(s_rmse)
        results["time_mean"].append(m_time)
        results["edges_mean"].append(m_edges)
        
        print(f"{m:<7} | {m_rmse:<12.4f} | {s_rmse:<10.4f} | {m_time:<10.4f} | {int(m_edges):<7}")

    # --- Plotting Logic ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_time = 'tab:blue'
    ax1.set_xlabel('Landmarks per User (M)')
    ax1.set_ylabel('Mean Processing Time (s)', color=color_time)
    ax1.plot(results["m"], results["time_mean"], 'o-', color=color_time, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_time)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_rmse = 'tab:red'
    ax2.set_ylabel('Mean RMSE (m) [Log Scale]', color=color_rmse)
    
    # Simple line plot for mean RMSE (no error bars)
    ax2.plot(results["m"], results["rmse_mean"], 's--', color=color_rmse, linewidth=2)
    
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color_rmse)

    plt.title(f'Multi-Trial Scaling: {n_users} Users ({n_trials} Trials per Point)')
    fig.tight_layout()
    
    plot_path = "results/salience_multi_trial.png"
    plt.savefig(plot_path)
    print(f"\nGraph saved to: {plot_path}")
    
    with open("results/salience_multitrial_data.csv", "w") as f:
        f.write("m_points,rmse_mean,rmse_std,time_mean,edges_mean\n")
        for i in range(len(results["m"])):
            f.write(f"{results['m'][i]},{results['rmse_mean'][i]},{results['rmse_std'][i]},{results['time_mean'][i]},{results['edges_mean'][i]}\n")

if __name__ == "__main__":
    run_high_res_salience_benchmark()
