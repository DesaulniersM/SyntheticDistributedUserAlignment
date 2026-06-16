import pandas as pd
import numpy as np

try:
    df = pd.read_csv("results/experiment_registry.csv")
    
    # Filter for the N values we are currently interested in
    target_ns = [80, 100, 120, 140, 160]
    df = df[df['n_users'].isin(target_ns)]
    
    # Filter for the sparsity frontier
    df = df[df['k_neighbors'].isin([2, 3, 4, 5])]
    
    # Filter for the two main solvers
    df = df[df['solver'].isin(['Spectral', 'GTSAM_Cold'])]
    
    if df.empty:
        print("No data available yet for these ranges.")
    else:
        # Calculate failure rate (RMSE > 1.0 is considered a failure)
        df['failure'] = df['mean_rmse'] > 1.0
        
        # Group by N, K, and Solver
        summary = df.groupby(['n_users', 'k_neighbors', 'solver']).agg(
            trials=('mean_rmse', 'count'),
            mean_rmse=('mean_rmse', lambda x: np.mean(x[x <= 1.0]) if any(x <= 1.0) else np.nan), # Mean of successful trials
            max_rmse=('mean_rmse', 'max'),
            failure_rate=('failure', 'mean')
        ).reset_index()
        
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        
except Exception as e:
    print(f"Error reading or processing data: {e}")
