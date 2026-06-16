from results.orchestrator import ExperimentOrchestrator
import os

def run_targeted_gtsam_study():
    orchestrator = ExperimentOrchestrator()
    
    # Study Parameters:
    # 1. Users (N): [10]
    # 2. Neighbors (K): [0, 1, 2, 3, 4, 5]
    # 3. Landmarks (M): 150
    # 4. Trials per cell: 3
    # 5. Solver: "GTSAM_Cold"
    
    print("Starting GTSAM Cold study for N=10...")
    orchestrator.run_master_study(
        n_users_list=[10],
        k_list=[0, 1, 2, 3, 4, 5],
        solvers=["GTSAM_Cold"],
        m_points=150,
        trials=3
    )
    print("Study complete. Results appended to results/experiment_registry.csv")

if __name__ == "__main__":
    run_targeted_gtsam_study()
