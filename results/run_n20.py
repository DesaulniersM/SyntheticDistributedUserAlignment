import os
import sys
from orchestrator import ExperimentOrchestrator

def run_specific_study():
    orchestrator = ExperimentOrchestrator()
    # Run only Spectral for N=20 across full K range
    orchestrator.run_master_study(
        n_users_list=[20],
        k_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        solvers=["Spectral"],
        m_points=150
    )

if __name__ == "__main__":
    run_specific_study()
