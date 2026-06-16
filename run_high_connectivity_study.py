import os
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')))
from orchestrator import ExperimentOrchestrator

def main():
    orchestrator = ExperimentOrchestrator()
    orchestrator.run_master_study(
        n_users_list=[30], 
        k_list=[15, 20], 
        solvers=["Ceres_Cold", "Ceres_Warm", "Spectral"], 
        m_points=150, 
        trials=1
    )

if __name__ == "__main__":
    main()
