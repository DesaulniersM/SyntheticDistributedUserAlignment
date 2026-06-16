import os
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')))
from orchestrator import ExperimentOrchestrator

def main():
    orchestrator = ExperimentOrchestrator()
    
    # Target N values from user request and existing partial data
    n_list = [80, 100, 120, 140, 160, 180, 200]
    
    # Full K sweep
    k_list = list(range(11)) # 0 to 10
    
    # Solvers to compare
    solvers = ["Spectral", "GTSAM_Cold"]
    
    print(f"Starting final sweep for N={n_list}")
    orchestrator.run_master_study(
        n_users_list=n_list,
        k_list=k_list,
        solvers=solvers,
        m_points=150,
        trials=3
    )
    print("Sweep complete. Generating updated heatmaps...")
    
    # Run plotting scripts
    import subprocess
    subprocess.run([sys.executable, "results/plot_heatmap.py"])
    subprocess.run([sys.executable, "results/plot_gtsam_heatmap.py"])

if __name__ == "__main__":
    main()
