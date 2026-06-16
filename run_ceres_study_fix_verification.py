
import os
import sys
import pandas as pd
from results.orchestrator import ExperimentOrchestrator

def main():
    orchestrator = ExperimentOrchestrator()
    
    n_users_list = [10, 30]
    k_list = [3, 6]
    solvers = ["Spectral", "Ceres_Cold", "Ceres_Warm"]
    m_points = 150
    trials = 1
    
    print("Starting Targeted Ceres Comparison Study...")
    orchestrator.run_master_study(
        n_users_list=n_users_list,
        k_list=k_list,
        solvers=solvers,
        m_points=m_points,
        trials=trials
    )
    
    print("\nStudy Complete. Results in results/experiment_registry.csv")
    
    # Extract and display the requested comparison table
    df = orchestrator.registry
    # Filter for the current study parameters to be sure
    mask = (
        df['n_users'].isin(n_users_list) & 
        df['k_neighbors'].isin(k_list) & 
        df['m_points'] == m_points &
        df['solver'].isin(solvers)
    )
    study_results = df[mask].copy()
    
    # Pivot for direct comparison
    pivot_df = study_results.pivot_table(
        index=['n_users', 'k_neighbors'], 
        columns='solver', 
        values='mean_rmse'
    )
    
    print("\nRMSE Comparison Table:")
    print(pivot_df.to_string())

if __name__ == "__main__":
    main()
