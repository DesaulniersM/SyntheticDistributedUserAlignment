import sys
import os
import numpy as np
sys.path.append(os.path.abspath('results'))
sys.path.append(os.path.abspath('.'))
from stress_test_spectral import SpectralStressTester, LevenbergMarquardtAlignmentManager
from common.conventions import get_yaw_from_matrix

class DebugLMManager(LevenbergMarquardtAlignmentManager):
    def compute_lm_global_alignment(self, anchor_id: int, anchor_pose: np.ndarray, initialization='cold', gt_poses=None):
        n_users = len(self.user_ids)
        id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        
        # 2. Initialization Strategies
        if initialization == 'cold':
            x0 = np.zeros(n_users * 4)
        elif initialization == 'warm_gt' and gt_poses is not None:
            x0 = np.zeros(n_users * 4)
            for uid, gt_t in gt_poses.items():
                idx = id_to_idx[uid]
                x0[idx*4 : idx*4+3] = gt_t[:3, 3] + np.random.normal(0, 0.5, 3)
                x0[idx*4+3] = get_yaw_from_matrix(gt_t) + np.random.normal(0, 0.1)
        else:
            x0 = np.zeros(n_users * 4)

        # Print x0 error
        x0_errs = []
        for uid, gt_t in gt_poses.items():
            idx = id_to_idx[uid]
            err = np.linalg.norm(x0[idx*4 : idx*4+3] - gt_t[:3, 3])
            x0_errs.append(err)
        print(f"Initial x0 RMSE: {np.mean(x0_errs):.4f}")

        def residual_func(x):
            residuals = []
            for (i, j), T_ij in self.edge_transforms.items():
                if T_ij is None: continue
                idx_i, idx_j = id_to_idx[i], id_to_idx[j]
                p_i, yaw_i = x[idx_i*4 : idx_i*4+3], x[idx_i*4+3]
                p_j, yaw_j = x[idx_j*4 : idx_j*4+3], x[idx_j*4+3]
                yaw_ij_obs = get_yaw_from_matrix(T_ij)
                t_ij_obs = T_ij[:3, 3]
                r_yaw = np.arctan2(np.sin((yaw_j - yaw_i) - yaw_ij_obs), np.cos((yaw_j - yaw_i) - yaw_ij_obs))
                c, s = np.cos(yaw_i), np.sin(yaw_i)
                R_i_inv = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
                t_ij_pred = R_i_inv @ (p_j - p_i)
                r_t = t_ij_pred - t_ij_obs
                w = self.edge_weights.get((i, j), 1.0)
                residuals.extend([r_yaw * w, r_t[0] * w, r_t[1] * w, r_t[2] * w])
            
            idx_a = id_to_idx[anchor_id]
            residuals.extend((x[idx_a*4 : idx_a*4+3] - anchor_pose[:3, 3]) * 1e6)
            anchor_yaw_gt = get_yaw_from_matrix(anchor_pose)
            r_yaw_anchor = np.arctan2(np.sin(x[idx_a*4+3] - anchor_yaw_gt), np.cos(x[idx_a*4+3] - anchor_yaw_gt))
            residuals.append(r_yaw_anchor * 1e6)
            return np.array(residuals)

        from scipy.optimize import least_squares
        res = least_squares(residual_func, x0, method='lm', xtol=1e-4)
        print(f"Optimization success: {res.success}, message: {res.message}")
        
        for uid in self.user_ids:
            idx = id_to_idx[uid]
            p, yaw = res.x[idx*4 : idx*4+3], res.x[idx*4+3]
            c, s = np.cos(yaw), np.sin(yaw)
            T = np.eye(4)
            T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            T[:3, 3] = p
            self.global_transforms[uid] = T

class DebugLMManager(LevenbergMarquardtAlignmentManager):
# ... existing code ...
import stress_test_spectral
stress_test_spectral.LevenbergMarquardtAlignmentManager = DebugLMManager
tester = SpectralStressTester(use_lm=True)
tester.generate_synthetic_data(n_users=10, topology='grid', n_landmarks=150, outlier_ratio=0.0, noise_std=0.0)
res = tester.run_test(neighbors_per_node=6, lm_init='warm_gt')
print("Final RMSE:", res['mean_rmse'])
