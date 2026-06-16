import os
import pandas as pd
import numpy as np
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from threading import Lock

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stress_test_spectral import SpectralStressTester

REGISTRY_PATH = "results/standardized_study.csv"
MASTER_WORLD_PATH = "results/world_geometry_large.npy"

# Global lock for thread-safe registry writes
file_lock = Lock()

def run_single_trial(n_users, k, s_type, m_actual, landmarks, master_poses, outliers, noise):
    """Worker function for parallel trials"""
    use_lm = (s_type == "LM_Cold" or s_type == "LM_Warm")
    use_gtsam = (s_type == "GTSAM_Cold" or s_type == "GTSAM_Warm")
    use_ceres = (s_type == "Ceres_Cold" or s_type == "Ceres_Warm")
    
    # Initialize tester inside process
    tester = SpectralStressTester(use_lm=use_lm, use_gtsam=use_gtsam, use_ceres=use_ceres)
    
    tester.load_master_data(
        master_poses,
        landmarks,
        n_users=n_users,
        outlier_ratio=outliers,
        noise_std=noise,
        max_visibility_dist=10.0
    )
    
    if s_type == "LM_Warm" or s_type == "GTSAM_Warm" or s_type == "Ceres_Warm":
        init_mode = 'warm_gt'
    elif s_type == "LM_Cold" or s_type == "GTSAM_Cold" or s_type == "Ceres_Cold":
        init_mode = 'cold'
    else:
        init_mode = None

    res = tester.run_test(neighbors_per_node=k, lm_init=init_mode)
    
    return {
        "timestamp": time.strftime("%H:%M:%S"),
        "solver": s_type,
        "n_users": n_users,
        "k_neighbors": k,
        "m_points": m_actual,
        "noise_std": noise,
        "outlier_ratio": outliers,
        "mean_rmse": res["mean_rmse"],
        "duration": res["duration"],
        "edges": res["edges_count"],
        "stage2_iters": res.get("stage2_iterations", 0),
        "stage3_iters": res.get("stage3_iterations", 0)
    }

class ExperimentOrchestrator:
    def __init__(self, world_path=MASTER_WORLD_PATH, registry_path=REGISTRY_PATH):
        self.registry_path = registry_path
        self.registry = self._load_registry()
        self.master_world = self._load_master_world(world_path)

    def _load_master_world(self, world_path):
        if os.path.exists(world_path):
            print(f"Loading master world from {world_path}")
            return np.load(world_path, allow_pickle=True).item()
        return None

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            return pd.read_csv(self.registry_path)
        else:
            return pd.DataFrame(columns=[
                "timestamp", "solver", "n_users", "k_neighbors", "m_points", 
                "noise_std", "outlier_ratio", "mean_rmse", "duration", "edges",
                "stage2_iters", "stage3_iters"
            ])

    def save_trial(self, data: dict):
        with file_lock:
            # Reload registry to ensure we don't overwrite other processes' work
            self.registry = self._load_registry()
            self.registry = pd.concat([self.registry, pd.DataFrame([data])], ignore_index=True)
            self.registry.to_csv(self.registry_path, index=False)

    def run_master_study(self, 
                         n_users_list: list = [20, 40, 60, 80],
                         k_list: list = [2, 3, 4, 5, 8],
                         solvers: list = ["Spectral", "LM_Cold"],
                         m_points: int = None,
                         trials: int = 1,
                         max_workers: int = None):
        if self.master_world is None:
            print("Error: Master world not loaded. Run generate_master_world.py first.")
            return

        # Auto-detect workers if not specified
        if max_workers is None:
            max_workers = os.cpu_count() or 1

        noise = 0.02
        outliers = 0.1
        
        landmarks_all = self.master_world['landmarks']
        m_actual = m_points or len(landmarks_all)
        
        if m_points is not None and m_points < len(landmarks_all):
            np.random.seed(42)
            idx = np.random.choice(len(landmarks_all), m_points, replace=False)
            landmarks = landmarks_all[idx]
        else:
            landmarks = landmarks_all

        print(f"\n>>> Running Master World Study ({'Parallel: ' + str(max_workers) if max_workers > 1 else 'Sequential'})")
        print(f"Sweep: N={n_users_list}, K={k_list}, Solvers={solvers}, M={m_actual}, Trials={trials}")
        print("-" * 75)

        tasks = []
        for n_users in n_users_list:
            for k in k_list:
                for s_type in solvers:
                    if not self.registry.empty:
                        mask = (
                            (self.registry['n_users'] == n_users) & 
                            (self.registry['k_neighbors'] == k) & 
                            (self.registry['solver'] == s_type) &
                            (self.registry['m_points'] == m_actual)
                        )
                        count = len(self.registry[mask])
                    else:
                        count = 0
                        
                    needed = trials - count
                    if needed <= 0:
                        continue
                        
                    print(f"Queueing N={n_users:<2} | K={k:<2} | {s_type:<10} | {needed} trials")
                    for _ in range(needed):
                        tasks.append((n_users, k, s_type))

        if not tasks:
            print("No new trials needed.")
            return

        if max_workers > 1:
            # Parallel Path
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_single_trial, t[0], t[1], t[2], m_actual, landmarks, 
                                        self.master_world['user_poses'], outliers, noise) for t in tasks]
                for future in futures:
                    try:
                        res_data = future.result()
                        self.save_trial(res_data)
                        print(f"  Finished: N={res_data['n_users']:<2} | K={res_data['k_neighbors']:<2} | {res_data['solver']:<10} | RMSE: {res_data['mean_rmse']:.4f}m")
                    except Exception as e:
                        print(f"  Trial failed with error: {e}")
        else:
            # Sequential Path (Safe for 1-core machines)
            for t in tasks:
                try:
                    res_data = run_single_trial(t[0], t[1], t[2], m_actual, landmarks, 
                                              self.master_world['user_poses'], outliers, noise)
                    self.save_trial(res_data)
                    print(f"  Finished: N={res_data['n_users']:<2} | K={res_data['k_neighbors']:<2} | {res_data['solver']:<10} | RMSE: {res_data['mean_rmse']:.4f}m")
                except Exception as e:
                    print(f"  Trial failed with error: {e}")
        
        print(f"Study Complete.")

    def run_robustness_study(self, 
                             n_users: int = 60, 
                             k_list: list = [2, 3, 4, 6, 8, 12], 
                             m_points: int = 150,
                             trials: int = 2):
        
        noise = 0.02
        outliers = 0.1
        solvers = ["Spectral", "LM_Cold"]
        
        print(f"\n>>> Running Global Robustness Study (N={n_users}, M={m_points})")
        print(f"Sweep: K={k_list}")
        print("-" * 65)

        for k in k_list:
            for t in range(trials):
                # 1. Generate ONE world for both solvers to compete on
                gen = SpectralStressTester(use_lm=False)
                gen.generate_synthetic_data(n_users=n_users, topology='grid', n_landmarks=m_points, 
                                          outlier_ratio=outliers, noise_std=noise, max_visibility_dist=10.0)
                
                for s_type in solvers:
                    # 2. Setup the specific solver
                    use_lm = (s_type == "LM_Cold")
                    tester = SpectralStressTester(use_lm=use_lm)
                    
                    # 3. Synchronize Data
                    tester.manager.user_ids = list(gen.manager.user_ids)
                    tester.manager.user_clouds = gen.manager.user_clouds.copy()
                    tester.manager.user_features = gen.manager.user_features.copy()
                    tester.manager.user_gravities = gen.manager.user_gravities.copy()
                    tester.gt_poses = gen.gt_poses
                    
                    # 4. Run Test
                    init_mode = 'cold' if use_lm else None
                    res = tester.run_test(neighbors_per_node=k, lm_init=init_mode)
                    
                    # 5. Log
                    self.save_trial({
                        "timestamp": time.strftime("%H:%M:%S"),
                        "solver": s_type,
                        "n_users": n_users,
                        "k_neighbors": k,
                        "m_points": m_points,
                        "noise_std": noise,
                        "outlier_ratio": outliers,
                        "mean_rmse": res["mean_rmse"],
                        "duration": res["duration"],
                        "edges": res["edges_count"]
                    })
                    print(f"Trial {t+1} | K={k:<2} | {s_type:<10} | RMSE: {res['mean_rmse']:.4f}m")

    def run_scale_invariance_study(self, 
                                   n_users_list: list = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
                                   k: int = 3,
                                   solvers: list = ["Spectral", "GTSAM_Cold"],
                                   trials: int = 3):
        if self.master_world is None:
            print("Error: Master world not loaded.")
            return

        noise = 0.02
        outliers = 0.1
        landmarks = self.master_world['landmarks']
        m_actual = len(landmarks)

        print(f"\n>>> Running Scale Invariance Study")
        print(f"Sweep: N={n_users_list}, K={k}, Solvers={solvers}, M={m_actual}, Trials={trials}")
        print("-" * 75)

        for n_users in n_users_list:
            for s_type in solvers:
                # Count existing trials
                if not self.registry.empty:
                    mask = (
                        (self.registry['n_users'] == n_users) & 
                        (self.registry['k_neighbors'] == k) & 
                        (self.registry['solver'] == s_type) &
                        (self.registry['m_points'] == m_actual)
                    )
                    count = len(self.registry[mask])
                else:
                    count = 0
                
                needed = trials - count
                if needed <= 0:
                    continue
                
                print(f"N={n_users:<3} | K={k} | {s_type:<10} | Running {needed} trials...")
                
                for t in range(needed):
                    use_gtsam = (s_type == "GTSAM_Cold")
                    tester = SpectralStressTester(use_lm=False, use_gtsam=use_gtsam)
                    
                    tester.load_master_data(
                        self.master_world['user_poses'],
                        landmarks,
                        n_users=n_users,
                        outlier_ratio=outliers,
                        noise_std=noise,
                        max_visibility_dist=10.0
                    )
                    
                    init_mode = 'cold' if use_gtsam else None
                    res = tester.run_test(neighbors_per_node=k, lm_init=init_mode)
                    
                    self.save_trial({
                        "timestamp": time.strftime("%H:%M:%S"),
                        "solver": s_type,
                        "n_users": n_users,
                        "k_neighbors": k,
                        "m_points": m_actual,
                        "noise_std": noise,
                        "outlier_ratio": outliers,
                        "mean_rmse": res["mean_rmse"],
                        "duration": res["duration"],
                        "edges": res["edges_count"]
                    })
                    print(f"  Trial {count + t + 1}/{trials} | RMSE: {res['mean_rmse']:.4f}m | Dur: {res['duration']:.2f}s")
            print(f"Finished N={n_users} group...")

        print("\n>>> Scale Invariance Study Complete.")

if __name__ == "__main__":
    # If this is run directly, it will perform the high-scale study
    orchestrator = ExperimentOrchestrator(world_path="results/world_geometry_large.npy")
    orchestrator.run_scale_invariance_study(
        n_users_list=[10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
        k=3,
        solvers=["Spectral", "GTSAM_Cold"],
        trials=3
    )
