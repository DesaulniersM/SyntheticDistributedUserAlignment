import os
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')))
from orchestrator import ExperimentOrchestrator

def main():
    orchestrator = ExperimentOrchestrator()
    
    from orchestrator import MASTER_WORLD_PATH, REGISTRY_PATH
    
    # User-requested test subset
    n_list = [20, 140]
    k_list = list(range(11)) # 0 to 10
    solvers = ["Spectral", "GTSAM_Cold"]
    
    print(f"Starting TEST SUBSET for N={n_list}")
    print(f"World: {MASTER_WORLD_PATH}")
    print(f"Registry: {REGISTRY_PATH}")
    
    orchestrator.run_master_study(
        n_users_list=n_list,
        k_list=k_list,
        solvers=solvers,
        m_points=150,
        trials=3
    )
    
    print("Test subset complete. Regenerating interim visuals...")
    import subprocess
    subprocess.run([sys.executable, "results/plot_heatmap.py"])
    subprocess.run([sys.executable, "results/plot_gtsam_heatmap.py"])

if __name__ == "__main__":
    main()
