import sys
import os
sys.path.append(os.path.abspath('results'))
from orchestrator import ExperimentOrchestrator

if __name__ == "__main__":
    orchestrator = ExperimentOrchestrator()
    orchestrator.run_master_study(
        n_users_list=[10, 20, 60],
        k_list=[3, 6, 10],
        solvers=["LM_Warm"],
        m_points=150,
        trials=1
    )
