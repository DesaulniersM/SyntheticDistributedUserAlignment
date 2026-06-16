import os
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')))
from orchestrator import ExperimentOrchestrator

def main():
    orchestrator = ExperimentOrchestrator()
    
    # Full N list for a complete standardized study
    # We include N=10 to 60 (which need rerunning) 
    # and N=80 to 200 (which are already in the large_world_study.csv and will be skipped)
    n_list = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200]
    
    # Full K sweep
    k_list = list(range(11)) # 0 to 10
    
    # Solvers
    solvers = ["Spectral", "GTSAM_Cold"]
    
    print(f"Starting standardized Large World study for N={n_list}")
    orchestrator.run_master_study(
        n_users_list=n_list,
        k_list=k_list,
        solvers=solvers,
        m_points=150,
        trials=3
    )
    
    print("Study complete. Regenerating heatmaps...")
    import subprocess
    # Note: We need to update the plotting scripts to use the new CSV
    subprocess.run([sys.executable, "results/plot_heatmap.py"])
    subprocess.run([sys.executable, "results/plot_gtsam_heatmap.py"])

if __name__ == "__main__":
    main()
