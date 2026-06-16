import os
import sys
import pandas as pd
import numpy as np
from results.orchestrator import ExperimentOrchestrator

# Constants for the Latency Study
NETWORK_LATENCY_MS = 100  # Latency per hop (communication round) in milliseconds
SUBSET_N = [20, 60, 100]
SUBSET_K = [3, 6, 10]
SOLVERS = ["Spectral", "GTSAM_Warm"]
TRIALS = 10
REGISTRY_PATH = "results/network_latency_study.csv"

def run_latency_study():
    print(f"\n>>> Starting Network Latency Simulation Study")
    print(f"Parameters: Latency={NETWORK_LATENCY_MS}ms, N={SUBSET_N}, K={SUBSET_K}")
    print(f"Results will be saved to: {REGISTRY_PATH}")
    print("-" * 75)
    
    orchestrator = ExperimentOrchestrator(registry_path=REGISTRY_PATH)
    
    # 1. Execute the subset study with instrumented iteration counts
    orchestrator.run_master_study(
        n_users_list=SUBSET_N,
        k_list=SUBSET_K,
        solvers=SOLVERS,
        m_points=150,
        trials=TRIALS
    )
    
    # 2. Analyze and calculate 'System Latency'
    df = pd.read_csv(REGISTRY_PATH)
    
    # Calculation Logic:
    # Spectral: 
    #   Stage 2 (Local): Handled in parallel, but sequential for this study's proxy
    #   Stage 3 (Global): stage3_iters rounds of communication
    # GTSAM:
    #   Stage 3 (Global): stage3_iters rounds of communication (Levenberg-Marquardt steps)
    
    hop_s = NETWORK_LATENCY_MS / 1000.0
    
    df['comm_latency'] = df['stage3_iters'] * hop_s
    df['system_latency'] = df['duration'] + df['comm_latency']
    
    # Save the calculated metrics back to the unique CSV
    df.to_csv(REGISTRY_PATH, index=False)
    
    print("\n>>> Latency Analysis Complete.")
    summary = df.groupby(['solver', 'n_users']).agg({
        'mean_rmse': 'mean',
        'duration': 'mean',
        'stage3_iters': 'mean',
        'system_latency': 'mean'
    })
    print("\nSummary Statistics (Mean values):")
    print(summary.to_string())
    
    print(f"\nKey Finding:")
    for n in SUBSET_N:
        s_lat = df[(df['solver'] == 'Spectral') & (df['n_users'] == n)]['system_latency'].mean()
        g_lat = df[(df['solver'] == 'GTSAM_Warm') & (df['n_users'] == n)]['system_latency'].mean()
        speedup = g_lat / s_lat if s_lat > 0 else 0
        print(f"N={n}: Spectral is {speedup:.2f}x faster in system-time than GTSAM_Warm")

if __name__ == "__main__":
    run_latency_study()
