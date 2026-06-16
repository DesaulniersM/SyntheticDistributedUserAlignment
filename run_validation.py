from results.orchestrator import ExperimentOrchestrator

if __name__ == "__main__":
    orchestrator = ExperimentOrchestrator()
    # Execute validation study
    # N: [10, 30]
    # K: [3, 6]
    # M: 150
    # Solver: "Spectral"
    # Trials: 1 (implicit in run_master_study loop)
    orchestrator.run_master_study(
        n_users_list=[10, 30],
        k_list=[3, 6],
        solvers=["Spectral"],
        m_points=150
    )
