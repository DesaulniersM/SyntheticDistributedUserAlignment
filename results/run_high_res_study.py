import os
import pandas as pd
import numpy as np
import time
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from results.orchestrator import ExperimentOrchestrator
from stress_test_spectral import SpectralStressTester

class ResumableOrchestrator(ExperimentOrchestrator):
    def run_resumable_study(self, 
                            n_users_list: list,
                            k_list: list,
                            solvers: list,
                            m_points: int):
        if self.master_world is None:
            print("Error: Master world not found. Run generate_master_world.py first.")
            return

        noise = 0.02
        outliers = 0.1
        
        landmarks = self.master_world['landmarks']
        if m_points is not None and m_points < len(landmarks):
            # Deterministic subset for consistency
            np.random.seed(42)
            idx = np.random.choice(len(landmarks), m_points, replace=False)
            landmarks = landmarks[idx]

        print(f"\n>>> Running Resumable High-Res Connectivity Study")
        print(f"Sweep: N={n_users_list}, K={k_list}, Solvers={solvers}, M={m_points}")
        print("-" * 75)

        # Check existing trials
        existing_trials = set()
        if not self.registry.empty:
            for _, row in self.registry.iterrows():
                # tuple of (solver, n_users, k_neighbors, m_points)
                # handle potential type differences
                existing_trials.add((str(row['solver']), int(row['n_users']), int(row['k_neighbors']), int(row['m_points'])))

        trials_run = 0

        for n_users in n_users_list:
            for k in k_list:
                for s_type in solvers:
                    trial_key = (s_type, n_users, k, m_points)
                    if trial_key in existing_trials:
                        print(f"Skipping N={n_users:<2} | K={k:<2} | {s_type:<10} (Already in registry)")
                        continue

                    use_lm = (s_type == "LM_Cold")
                    tester = SpectralStressTester(use_lm=use_lm)
                    
                    try:
                        tester.load_master_data(
                            self.master_world['user_poses'],
                            landmarks,
                            n_users=n_users,
                            outlier_ratio=outliers,
                            noise_std=noise,
                            max_visibility_dist=10.0
                        )
                        
                        init_mode = 'cold' if use_lm else None
                        res = tester.run_test(neighbors_per_node=k, lm_init=init_mode)
                        
                        self.save_trial({
                            "timestamp": time.strftime("%H:%M:%S"),
                            "solver": s_type,
                            "n_users": n_users,
                            "k_neighbors": k,
                            "m_points": len(landmarks),
                            "noise_std": noise,
                            "outlier_ratio": outliers,
                            "mean_rmse": res["mean_rmse"],
                            "duration": res["duration"],
                            "edges": res["edges_count"]
                        })
                        print(f"N={n_users:<2} | K={k:<2} | {s_type:<10} | RMSE: {res['mean_rmse']:.4f}m | Dur: {res['duration']:.2f}s")
                        trials_run += 1
                    except Exception as e:
                        print(f"Failed N={n_users:<2} | K={k:<2} | {s_type:<10}: {e}")
                        # Depending on the exception, maybe k=0 is expected to fail or we want to record NaN
                        self.save_trial({
                            "timestamp": time.strftime("%H:%M:%S"),
                            "solver": s_type,
                            "n_users": n_users,
                            "k_neighbors": k,
                            "m_points": len(landmarks),
                            "noise_std": noise,
                            "outlier_ratio": outliers,
                            "mean_rmse": np.nan,
                            "duration": np.nan,
                            "edges": 0
                        })
                        print(f"Recorded failure in registry for N={n_users}, K={k}")

        print(f"\n>>> Study complete. {trials_run} new trials executed.")

if __name__ == "__main__":
    orchestrator = ResumableOrchestrator()
    n_users_list = [10, 30, 60]
    k_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    solvers = ["Spectral"]
    m_points = 150
    
    orchestrator.run_resumable_study(n_users_list=n_users_list, k_list=k_list, solvers=solvers, m_points=m_points)
