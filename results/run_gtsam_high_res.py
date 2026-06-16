import os
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from results.orchestrator import ExperimentOrchestrator

def main():
    orchestrator = ExperimentOrchestrator()
    
    n_users_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    k_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    solvers = ["GTSAM_Cold"]
    m_points = 150
    trials_per_cell = 3
    
    print(f"Starting GTSAM High-Res Study: N={n_users_list}, K={k_list}, M={m_points}, Trials={trials_per_cell}")
    
    orchestrator.run_master_study(
        n_users_list=n_users_list,
        k_list=k_list,
        solvers=solvers,
        m_points=m_points,
        trials=trials_per_cell
    )
    
    print("\nStudy Complete!")

if __name__ == "__main__":
    main()
