import sys
import os
sys.path.append(os.path.abspath('results'))
from orchestrator import ExperimentOrchestrator

if __name__ == "__main__":
    orchestrator = ExperimentOrchestrator()
    orchestrator.run_master_study(
        n_users_list=[10, 20],
        k_list=[3, 6],
        solvers=["LM_Cold"],
        m_points=150,
        trials=1
    )
