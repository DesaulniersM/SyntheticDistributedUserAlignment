import os
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')))
from orchestrator import ExperimentOrchestrator

def main():
    orchestrator = ExperimentOrchestrator()
    
    from orchestrator import MASTER_WORLD_PATH, REGISTRY_PATH
    
    # Uniform, standardized N list
    n_list = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    
    # Full K sweep
    k_list = list(range(11)) # 0 to 10
    
    # Main Solvers
    solvers = ["Spectral", "GTSAM_Cold"]
    
    print(f"Starting CLEAN STANDARDIZED study for N={n_list}")
    print(f"World: {MASTER_WORLD_PATH}")
    print(f"Registry: {REGISTRY_PATH}")
    
    orchestrator.run_master_study(
        n_users_list=n_list,
        k_list=k_list,
        solvers=solvers,
        m_points=150,
        trials=30
    )
    
    print("Standardized study complete. Regenerating final visuals...")
    import subprocess
    subprocess.run([sys.executable, "results/plot_heatmap.py"])
    subprocess.run([sys.executable, "results/plot_gtsam_heatmap.py"])
    subprocess.run([sys.executable, "results/plot_reliability.py"])

if __name__ == "__main__":
    main()
