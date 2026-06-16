
import os
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')))
from orchestrator import ExperimentOrchestrator

if __name__ == "__main__":
    orchestrator = ExperimentOrchestrator()
    orchestrator.run_master_study(
        n_users_list=[10, 20, 60],
        k_list=[10],
        solvers=["LM_Cold"],
        m_points=150,
        trials=1
    )
