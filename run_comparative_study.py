import os
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')))
from orchestrator import ExperimentOrchestrator

if __name__ == "__main__":
    orchestrator = ExperimentOrchestrator()
    
    # Run the study as requested
    # N: [10, 20, 60]
    # K: [3, 6, 10]
    # M: 150
    # Trials: 1
    # Solvers: Spectral, GTSAM_Cold, GTSAM_Warm
    
    orchestrator.run_master_study(
        n_users_list=[10, 20, 60],
        k_list=[3, 6, 10],
        solvers=["Spectral", "GTSAM_Cold", "GTSAM_Warm"],
        m_points=150,
        trials=1
    )
