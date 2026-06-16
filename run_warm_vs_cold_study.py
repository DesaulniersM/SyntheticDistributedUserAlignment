import sys
import os
import time

# Ensure we can find the project modules
sys.path.append(os.path.abspath('results'))
from orchestrator import ExperimentOrchestrator

def run_comparison():
    # 1. Setup a dedicated registry for this study
    new_registry = "results/warm_start_comparison.csv"
    print(f"Initializing Warm vs. Cold study. Data will be saved to: {new_registry}")
    
    orchestrator = ExperimentOrchestrator(registry_path=new_registry)
    
    # 2. Define the parameters
    # We'll use a representative sweep of N and K
    n_users = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    k_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # We compare:
    # - Spectral: Always Cold Start (The challenger)
    # - GTSAM_Warm: Initialized with Noisy GT (The baseline with a "head start")
    solvers = ["Spectral", "GTSAM_Warm"]
    
    trials = 30 # Run 30 trials per cell to get full statistics
    
    # 3. Execute
    orchestrator.run_master_study(
        n_users_list=n_users,
        k_list=k_neighbors,
        solvers=solvers,
        m_points=150,
        trials=trials
    )

if __name__ == "__main__":
    run_comparison()
