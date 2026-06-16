import os
import sys
import pandas as pd

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from results.orchestrator import ExperimentOrchestrator

def main():
    orchestrator = ExperimentOrchestrator()
    
    # Study Parameters:
    # Users (N): [10, 30, 60]
    # Neighbors (K): [0, 1, 2]
    # Landmarks (M): 150
    # Trials per cell: 1
    # Solver: "Spectral" only.
    
    n_users_list = [10, 30, 60]
    k_list = [0, 1, 2]
    solvers = ["Spectral"]
    m_points = 150
    
    print(f"Executing Connectivity Sanity Check...")
    print(f"N: {n_users_list}")
    print(f"K: {k_list}")
    print(f"M: {m_points}")
    print(f"Solvers: {solvers}")
    
    orchestrator.run_master_study(
        n_users_list=n_users_list,
        k_list=k_list,
        solvers=solvers,
        m_points=m_points
    )
    
    # Read results from registry for reporting
    df = pd.read_csv("results/experiment_registry.csv")
    
    # Filter for this specific run (roughly, by timestamp or just taking the last 9 rows)
    # Since we just ran 3*3=9 trials
    results = df.tail(len(n_users_list) * len(k_list) * len(solvers))
    
    print("\n>>> Combined Table of Results")
    print(results[["n_users", "k_neighbors", "mean_rmse", "edges"]].to_string(index=False))

if __name__ == "__main__":
    main()
