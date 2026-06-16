from orchestrator import ExperimentOrchestrator

def run_ceres_study():
    orchestrator = ExperimentOrchestrator()
    # Execute the requested 2x2 comparison study
    orchestrator.run_master_study(
        n_users_list=[10, 30],
        k_list=[3, 6],
        solvers=["Ceres_Cold", "Ceres_Warm"],
        m_points=150,
        trials=1
    )

if __name__ == "__main__":
    run_ceres_study()
