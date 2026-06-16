from results.orchestrator import ExperimentOrchestrator

if __name__ == "__main__":
    orchestrator = ExperimentOrchestrator()
    
    # Study Parameters:
    # 1. Users (N): [10, 30]
    # 2. Neighbors (K): [3, 6]
    # 3. Landmarks (M): 150
    # 4. Trials per cell: 1
    # 5. Solvers: "GTSAM_Cold" and "GTSAM_Warm"
    
    orchestrator.run_master_study(
        n_users_list=[10, 30],
        k_list=[3, 6],
        solvers=["GTSAM_Cold", "GTSAM_Warm"],
        m_points=150,
        trials=1
    )
