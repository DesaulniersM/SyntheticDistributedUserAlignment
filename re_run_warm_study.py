import os
import pandas as pd
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath('results'))
from orchestrator import ExperimentOrchestrator

REGISTRY_PATH = "results/experiment_registry.csv"

def clear_existing_entries(n_list, k_list, solver, m_points):
    if not os.path.exists(REGISTRY_PATH):
        return
    
    df = pd.read_csv(REGISTRY_PATH)
    initial_len = len(df)
    
    # Filter out the entries we want to re-run
    mask = (
        (df['solver'] == solver) &
        (df['n_users'].isin(n_list)) &
        (df['k_neighbors'].isin(k_list)) &
        (df['m_points'] == m_points)
    )
    
    df = df[~mask]
    
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} existing entries from registry.")
        df.to_csv(REGISTRY_PATH, index=False)
    else:
        print("No matching entries found to remove.")

if __name__ == "__main__":
    n_users = [10, 20]
    k_neighbors = [6, 10]
    solver = "LM_Warm"
    m_points = 150
    
    print(f"Clearing registry for {solver}, N={n_users}, K={k_neighbors}, M={m_points}...")
    clear_existing_entries(n_users, k_neighbors, solver, m_points)
    
    orchestrator = ExperimentOrchestrator()
    orchestrator.run_master_study(
        n_users_list=n_users,
        k_list=k_neighbors,
        solvers=[solver],
        m_points=m_points,
        trials=1
    )
    
    # After running, display the new results
    print("\n>>> New LM_Warm Results:")
    df = pd.read_csv(REGISTRY_PATH)
    new_results = df[
        (df['solver'] == solver) & 
        (df['n_users'].isin(n_users)) & 
        (df['k_neighbors'].isin(k_neighbors)) &
        (df['m_points'] == m_points)
    ].sort_values(['n_users', 'k_neighbors']).tail(4) # Get the last 4 runs if there were multiple
    
    # Actually, we cleared them, so they should be unique if trials=1.
    # But let's just grab the relevant ones.
    relevant = df[
        (df['solver'] == solver) & 
        (df['n_users'].isin(n_users)) & 
        (df['k_neighbors'].isin(k_neighbors)) &
        (df['m_points'] == m_points)
    ].sort_values(['n_users', 'k_neighbors'])
    
    print(relevant[['n_users', 'k_neighbors', 'mean_rmse', 'duration']].to_string(index=False))

    # Also get Spectral results for comparison
    print("\n>>> Spectral Comparison:")
    spectral = df[
        (df['solver'] == "Spectral") & 
        (df['n_users'].isin(n_users)) & 
        (df['k_neighbors'].isin(k_neighbors)) &
        (df['m_points'] == m_points)
    ].sort_values(['n_users', 'k_neighbors'])
    
    # Group by n_users, k_neighbors and take mean if multiple trials exist
    if not spectral.empty:
        spectral_summary = spectral.groupby(['n_users', 'k_neighbors'])['mean_rmse'].mean().reset_index()
        print(spectral_summary.to_string(index=False))
    else:
        print("No Spectral results found for comparison.")
