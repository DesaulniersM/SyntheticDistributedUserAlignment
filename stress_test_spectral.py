import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from solvers.global_solvers.MultiUserAlignment import SpectralAlignmentManager
from common.conventions import get_yaw_from_matrix
from scipy.optimize import least_squares
import gtsam
try:
    import pyceres
except ImportError:
    pyceres = None


class RansacSVDAlignmentManager(SpectralAlignmentManager):
    """
    Overrides the local solver to use RANSAC + SVD instead of 
    the Spatial Compatibility Matrix.
    """
    def _get_ransac_matches(self, src_id: int, tgt_id: int, n_iterations: int = 100, threshold: float = 0.1):
        # 1. Get initial mutual matches via features
        feat_src = self.user_features[src_id]
        feat_tgt = self.user_features[tgt_id]
        
        # Simple mutual matching on world-coord features
        dists = np.linalg.norm(feat_src[:, np.newaxis, :3] - feat_tgt[np.newaxis, :, :3], axis=2)
        idx_tgt = np.argmin(dists, axis=1)
        
        mutual_matches = []
        for i, j in enumerate(idx_tgt):
            if np.argmin(dists[:, j]) == i and dists[i, j] < 1.0:
                mutual_matches.append((i, j))
        
        if len(mutual_matches) < 4:
            return 0, []

        m_pts1 = self.user_clouds[src_id][[m[0] for m in mutual_matches]]
        m_pts2 = self.user_clouds[tgt_id][[m[1] for m in mutual_matches]]
        
        # 2. RANSAC Loop (Targeting 4-DoF: Yaw + Translation)
        best_inliers = []
        best_score = 0
        
        for _ in range(n_iterations):
            # Sample 3 points (minimum for rigid transform)
            idx = np.random.choice(len(mutual_matches), 3, replace=False)
            p1, p2 = m_pts1[idx], m_pts2[idx]
            
            # Simple SVD for 3D Transform
            c1, c2 = np.mean(p1, axis=0), np.mean(p2, axis=0)
            H = (p1 - c1).T @ (p2 - c2)
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[2,:] *= -1
                R = Vt.T @ U.T
            t = c1 - R @ c2
            
            # Count Inliers
            pred_p1 = (R @ m_pts2.T).T + t
            err = np.linalg.norm(m_pts1 - pred_p1, axis=1)
            inliers = [mutual_matches[i] for i in range(len(err)) if err[i] < threshold]
            
            if len(inliers) > best_score:
                best_score = len(inliers)
                best_inliers = inliers
        
        # Score is ratio of inliers (for Stage 3 weighting)
        final_score = best_score / len(mutual_matches)
        return final_score, best_inliers

    def compute_pairwise_transforms(self):
        """Uses RANSAC instead of Spectral Filtering"""
        for (i, j) in self.edge_transforms.keys():
            score, inliers = self._get_ransac_matches(i, j)
            
            if len(inliers) >= 4:
                # Calculate final SVD T_ij from all inliers
                p1 = self.user_clouds[i][[m[0] for m in inliers]]
                p2 = self.user_clouds[j][[m[1] for m in inliers]]
                
                c1, c2 = np.mean(p1, axis=0), np.mean(p2, axis=0)
                H = (p1 - c1).T @ (p2 - c2)
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                t = c1 - R @ c2
                
                T = np.eye(4)
                T[:3, :3] = R; T[:3, 3] = t
                self.edge_transforms[(i, j)] = T
                self.edge_weights[(i, j)] = score

class LevenbergMarquardtAlignmentManager(SpectralAlignmentManager):
    """
    Solves the Global Alignment problem using non-linear least squares (LM).
    Requires an initial guess for the poses.
    """
    def compute_lm_global_alignment(self, anchor_id: int, anchor_pose: np.ndarray, initialization='cold', gt_poses=None):
        n_users = len(self.user_ids)
        id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        
        # 2. Initialization Strategies
        if initialization == 'cold':
            x0 = np.zeros(n_users * 4)
        elif initialization == 'warm_gt' and gt_poses is not None:
            # GT + 0.5m/5deg noise to test "Basin of Attraction"
            x0 = np.zeros(n_users * 4)
            for uid, gt_t in gt_poses.items():
                idx = id_to_idx[uid]
                x0[idx*4 : idx*4+3] = gt_t[:3, 3] + np.random.normal(0, 0.5, 3)
                x0[idx*4+3] = get_yaw_from_matrix(gt_t) + np.random.normal(0, 0.1)
        elif initialization == 'warm_current':
            # Start from current estimates (Refinement mode)
            x0 = np.zeros(n_users * 4)
            for uid, T in self.global_transforms.items():
                idx = id_to_idx[uid]
                x0[idx*4 : idx*4+3] = T[:3, 3]
                x0[idx*4+3] = get_yaw_from_matrix(T)
        else:
            x0 = np.zeros(n_users * 4)

        def residual_func(x):
            residuals = []
            for (i, j), T_ij in self.edge_transforms.items():
                if T_ij is None: continue
                # Convention: T_i = T_j * T_ij. So swap j and i from the naive T_j = T_i * T_ij
                idx_j, idx_i = id_to_idx[i], id_to_idx[j]
                
                p_i, yaw_i = x[idx_i*4 : idx_i*4+3], x[idx_i*4+3]
                p_j, yaw_j = x[idx_j*4 : idx_j*4+3], x[idx_j*4+3]
                
                yaw_ij_obs = get_yaw_from_matrix(T_ij)
                t_ij_obs = T_ij[:3, 3]
                
                # Yaw residual
                r_yaw = np.arctan2(np.sin((yaw_j - yaw_i) - yaw_ij_obs), np.cos((yaw_j - yaw_i) - yaw_ij_obs))
                
                # Translation residual (T_j = T_i @ T_ij => t_ij = R_i^T @ (p_j - p_i))
                c, s = np.cos(yaw_i), np.sin(yaw_i)
                R_i_inv = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
                t_ij_pred = R_i_inv @ (p_j - p_i)
                r_t = t_ij_pred - t_ij_obs
                
                w = self.edge_weights.get((i, j), 1.0)
                residuals.extend([r_yaw * w, r_t[0] * w, r_t[1] * w, r_t[2] * w])
            
            # Anchor constraint: Hard-pin both position and rotation
            idx_a = id_to_idx[anchor_id]
            residuals.extend((x[idx_a*4 : idx_a*4+3] - anchor_pose[:3, 3]) * 1e6)
            # Ensure anchor yaw is also pinned with wrapping logic
            anchor_yaw_gt = get_yaw_from_matrix(anchor_pose)
            r_yaw_anchor = np.arctan2(np.sin(x[idx_a*4+3] - anchor_yaw_gt), np.cos(x[idx_a*4+3] - anchor_yaw_gt))
            residuals.append(r_yaw_anchor * 1e6)
            return np.array(residuals)

        res = least_squares(residual_func, x0, method='lm', xtol=1e-4)
        
        for uid in self.user_ids:
            idx = id_to_idx[uid]
            p, yaw = res.x[idx*4 : idx*4+3], res.x[idx*4+3]
            c, s = np.cos(yaw), np.sin(yaw)
            T = np.eye(4)
            T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            T[:3, 3] = p
            self.global_transforms[uid] = T

class GTSAMAlignmentManager(SpectralAlignmentManager):
    """
    Solves the Global Alignment problem using the industry-standard GTSAM library.
    Constructs a factor graph and uses Levenberg-Marquardt with robust Huber noise.
    """
    def compute_gtsam_global_alignment(self, anchor_id: int, anchor_pose: np.ndarray, initialization='cold', gt_poses=None):
        graph = gtsam.NonlinearFactorGraph()

        # 1. Add Prior Factor for the Anchor Node (Hard-pin)
        anchor_key = int(anchor_id)
        prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-4) # Stiff prior
        graph.add(gtsam.PriorFactorPose3(anchor_key, gtsam.Pose3(anchor_pose), prior_noise))

        # 2. Add BetweenFactors for each relative transform
        # Using Robust Huber model to handle outliers fairly
        base_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05]))
        robust_model = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.0), base_noise)

        for (i, j), T_ij in self.edge_transforms.items():
            if T_ij is None: continue
            # GTSAM Factor: T_i = T_j * T_ij  (Inverted from naive T_j = T_i * T_ij)
            graph.add(gtsam.BetweenFactorPose3(int(j), int(i), gtsam.Pose3(T_ij), robust_model))

        # 3. Initialization
        initial_values = gtsam.Values()
        if initialization == 'cold':
            for uid in self.user_ids:
                initial_values.insert(int(uid), gtsam.Pose3())
        elif initialization == 'warm_gt' and gt_poses is not None:
            for uid, gt_t in gt_poses.items():
                noise_t = gt_t[:3, 3] + np.random.normal(0, 0.5, 3)
                noise_yaw = get_yaw_from_matrix(gt_t) + np.random.normal(0, 0.1)
                c, s = np.cos(noise_yaw), np.sin(noise_yaw)
                R_noisy = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                T_noisy = np.eye(4)
                T_noisy[:3, :3] = R_noisy; T_noisy[:3, 3] = noise_t
                initial_values.insert(int(uid), gtsam.Pose3(T_noisy))
        else:
            for uid in self.user_ids:
                initial_values.insert(int(uid), gtsam.Pose3())

        # 4. Optimize
        try:
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
            result = optimizer.optimize()

            # 5. Store results
            for uid in self.user_ids:
                self.global_transforms[uid] = result.atPose3(int(uid)).matrix()
        except RuntimeError as e:
            print(f"  GTSAM Error: {e}")
            # Fallback: keep initial values or set to identity
            for uid in self.user_ids:
                self.global_transforms[uid] = initial_values.atPose3(int(uid)).matrix()
class CeresAlignmentManager(SpectralAlignmentManager):
    """
    Solves the Global Alignment problem using pyceres.
    Uses a 4-DoF pose representation (X, Y, Z, Yaw).
    """
    def compute_ceres_global_alignment(self, anchor_id: int, anchor_pose: np.ndarray, initialization='cold', gt_poses=None):
        n_users = len(self.user_ids)
        
        # 1. Initialize parameters (one 4-element array per user)
        params = {}
        for uid in self.user_ids:
            p = np.zeros(4)
            if initialization == 'cold':
                pass # Already zero
            elif initialization == 'warm_gt' and gt_poses is not None:
                gt_t = gt_poses[uid]
                p[:3] = gt_t[:3, 3] + np.random.normal(0, 0.5, 3)
                p[3] = get_yaw_from_matrix(gt_t) + np.random.normal(0, 0.1)
            elif initialization == 'warm_current':
                T = self.global_transforms.get(uid, np.eye(4))
                p[:3] = T[:3, 3]
                p[3] = get_yaw_from_matrix(T)
            params[uid] = p

        # 2. Define Cost Function
        class Pose4DoFCost(pyceres.CostFunction):
            def __init__(self, t_ij_obs, yaw_ij_obs, weight):
                super().__init__()
                self.set_num_residuals(4)
                self.set_parameter_block_sizes([4, 4])
                self.t_ij_obs = t_ij_obs
                self.yaw_ij_obs = yaw_ij_obs
                self.weight = weight

            def Evaluate(self, parameters, residuals, jacobians):
                p_i = parameters[0]
                p_j = parameters[1]
                
                x_i, y_i, z_i, yaw_i = p_i
                x_j, y_j, z_j, yaw_j = p_j
                
                c, s = np.cos(yaw_i), np.sin(yaw_i)
                
                # Yaw residual
                dyaw = yaw_j - yaw_i
                r_yaw = np.arctan2(np.sin(dyaw - self.yaw_ij_obs), np.cos(dyaw - self.yaw_ij_obs))
                
                # Translation residual
                dx, dy, dz = x_j - x_i, y_j - y_i, z_j - z_i
                r_x = c * dx + s * dy - self.t_ij_obs[0]
                r_y = -s * dx + c * dy - self.t_ij_obs[1]
                r_z = dz - self.t_ij_obs[2]
                
                residuals[0] = r_yaw * self.weight
                residuals[1] = r_x * self.weight
                residuals[2] = r_y * self.weight
                residuals[3] = r_z * self.weight
                
                if jacobians is not None:
                    if jacobians[0] is not None:
                        # Row 0: r_yaw w.r.t [x_i, y_i, z_i, yaw_i]
                        jacobians[0][0] = 0; jacobians[0][1] = 0; jacobians[0][2] = 0; jacobians[0][3] = -1.0 * self.weight
                        # Row 1: r_x w.r.t [x_i, y_i, z_i, yaw_i]
                        jacobians[0][4] = -c * self.weight
                        jacobians[0][5] = -s * self.weight
                        jacobians[0][6] = 0
                        jacobians[0][7] = (-s * dx + c * dy) * self.weight
                        # Row 2: r_y w.r.t [x_i, y_i, z_i, yaw_i]
                        jacobians[0][8] = s * self.weight
                        jacobians[0][9] = -c * self.weight
                        jacobians[0][10] = 0
                        jacobians[0][11] = (-c * dx - s * dy) * self.weight
                        # Row 3: r_z w.r.t [x_i, y_i, z_i, yaw_i]
                        jacobians[0][12] = 0; jacobians[0][13] = 0; jacobians[0][14] = -1.0 * self.weight; jacobians[0][15] = 0
                    
                    if jacobians[1] is not None:
                        # Row 0: r_yaw w.r.t [x_j, y_j, z_j, yaw_j]
                        jacobians[1][0] = 0; jacobians[1][1] = 0; jacobians[1][2] = 0; jacobians[1][3] = 1.0 * self.weight
                        # Row 1: r_x w.r.t [x_j, y_j, z_j, yaw_j]
                        jacobians[1][4] = c * self.weight
                        jacobians[1][5] = s * self.weight
                        jacobians[1][6] = 0
                        jacobians[1][7] = 0
                        # Row 2: r_y w.r.t [x_j, y_j, z_j, yaw_j]
                        jacobians[1][8] = -s * self.weight
                        jacobians[1][9] = c * self.weight
                        jacobians[1][10] = 0
                        jacobians[1][11] = 0
                        # Row 3: r_z w.r.t [x_j, y_j, z_j, yaw_j]
                        jacobians[1][12] = 0; jacobians[1][13] = 0; jacobians[1][14] = 1.0 * self.weight; jacobians[1][15] = 0
                
                return True

        # 3. Build Problem
        prob = pyceres.Problem()
        loss = pyceres.HuberLoss(1.0)
        
        for (i, j), T_ij in self.edge_transforms.items():
            if T_ij is None: continue
            yaw_ij_obs = get_yaw_from_matrix(T_ij)
            t_ij_obs = T_ij[:3, 3]
            w = self.edge_weights.get((i, j), 1.0)
            
            cost_func = Pose4DoFCost(t_ij_obs, yaw_ij_obs, w)
            # Swap i and j to match T_i = T_j * T_ij
            prob.add_residual_block(cost_func, loss, [params[j], params[i]])

        # 4. Pin Anchor
        anchor_yaw_gt = get_yaw_from_matrix(anchor_pose)
        params[anchor_id][:3] = anchor_pose[:3, 3]
        params[anchor_id][3] = anchor_yaw_gt
        prob.set_parameter_block_constant(params[anchor_id])

        # 5. Solve
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
        options.minimizer_progress_to_stdout = False
        options.max_num_iterations = 100
        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        
        # 6. Store results
        for uid in self.user_ids:
            p = params[uid]
            c, s = np.cos(p[3]), np.sin(p[3])
            T = np.eye(4)
            T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            T[:3, 3] = p[:3]
            self.global_transforms[uid] = T

class SpectralStressTester:
    def __init__(self, use_ransac=False, use_lm=False, use_gtsam=False, use_ceres=False):
        self.use_lm = use_lm
        self.use_ransac = use_ransac
        self.use_gtsam = use_gtsam
        self.use_ceres = use_ceres
        self._init_manager()
        self.gt_poses = {}

    def _init_manager(self):
        if self.use_ceres:
            self.manager = CeresAlignmentManager()
        elif self.use_gtsam:
            self.manager = GTSAMAlignmentManager()
        elif self.use_lm:
            self.manager = LevenbergMarquardtAlignmentManager()
        elif self.use_ransac:
            self.manager = RansacSVDAlignmentManager()
        else:
            self.manager = SpectralAlignmentManager()

    def load_master_data(self, user_poses, landmarks, n_users=10, outlier_ratio=0.1, noise_std=0.02, max_visibility_dist=10.0):
        """Loads data from the persistent world file instead of generating it."""
        self._init_manager()
        self.gt_poses = {i: user_poses[i] for i in range(n_users)}
        
        for i in range(n_users):
            T_world_local = self.gt_poses[i]
            T_local_world = np.linalg.inv(T_world_local)
            
            dists = np.linalg.norm(landmarks - T_world_local[:3, 3], axis=1)
            visible_mask = dists < max_visibility_dist
            
            pts_local = (T_local_world[:3, :3] @ landmarks[visible_mask].T).T + T_local_world[:3, 3]
            feats = np.zeros((len(pts_local), 32))
            feats[:, :3] = landmarks[visible_mask]
            
            if noise_std > 0 and len(pts_local) > 0:
                pts_local += np.random.normal(0, noise_std, pts_local.shape)
            
            if outlier_ratio > 0 and len(pts_local) > 0:
                n_out = int(len(pts_local) * outlier_ratio)
                out_idx = np.random.choice(len(pts_local), n_out, replace=False)
                pts_local[out_idx] = np.random.uniform(-max_visibility_dist, max_visibility_dist, (n_out, 3))
                feats[out_idx, :3] = np.random.uniform(-100, 100, (n_out, 3))

            self.manager.add_user_data(i, pts_local)
            self.manager.user_features[i] = feats
            self.manager.user_gravities[i] = np.array([0, 0, 1])

    def generate_synthetic_data(self, 
                                 n_users: int = 10, 
                                 topology: str = 'circle', 
                                 n_landmarks: int = 500,
                                 outlier_ratio: float = 0.0,
                                 noise_std: float = 0.0,
                                 max_visibility_dist: float = 5.0):
        """Generates synthetic users and landmark data."""
        self._init_manager()
        self.gt_poses = {}
        side = int(np.ceil(np.sqrt(n_users)))
        for i in range(n_users):
            if topology == 'circle':
                angle = (2 * np.pi * i) / n_users
                pos = np.array([5.0 * np.cos(angle), 5.0 * np.sin(angle), 0.0])
                yaw = angle + np.pi/2
            elif topology == 'chain':
                pos = np.array([2.0 * i, 0.0, 0.0])
                yaw = 0.0
            elif topology == 'grid':
                r, c = divmod(i, side)
                pos = np.array([r * 2.5, c * 2.5, 0.0])
                yaw = np.random.uniform(0, 2*np.pi)
            
            c_val, s_val = np.cos(yaw), np.sin(yaw)
            T = np.eye(4)
            T[:3, :3] = [[c_val, -s_val, 0], [s_val, c_val, 0], [0, 0, 1]]
            T[:3, 3] = pos
            self.gt_poses[i] = T

        all_pos = np.array([p[:3, 3] for p in self.gt_poses.values()])
        min_b, max_b = np.min(all_pos, axis=0) - 2.0, np.max(all_pos, axis=0) + 2.0
        world_landmarks = np.random.uniform(min_b, max_b, (n_landmarks, 3))

        for i in range(n_users):
            T_world_local = self.gt_poses[i]
            T_local_world = np.linalg.inv(T_world_local)
            dists = np.linalg.norm(world_landmarks - T_world_local[:3, 3], axis=1)
            visible_mask = dists < max_visibility_dist
            pts_local = (T_local_world[:3, :3] @ world_landmarks[visible_mask].T).T + T_local_world[:3, 3]
            feats = np.zeros((len(pts_local), 32))
            feats[:, :3] = world_landmarks[visible_mask]
            if noise_std > 0 and len(pts_local) > 0:
                pts_local += np.random.normal(0, noise_std, pts_local.shape)
            if outlier_ratio > 0 and len(pts_local) > 0:
                n_out = int(len(pts_local) * outlier_ratio)
                out_idx = np.random.choice(len(pts_local), n_out, replace=False)
                pts_local[out_idx] = np.random.uniform(-max_visibility_dist, max_visibility_dist, (n_out, 3))
                feats[out_idx, :3] = np.random.uniform(-100, 100, (n_out, 3))
            self.manager.add_user_data(i, pts_local)
            self.manager.user_features[i] = feats
            self.manager.user_gravities[i] = np.array([0, 0, 1])

    def run_test(self, neighbors_per_node: int = 3, use_irls: bool = True, lm_init='cold'):
        """Runs the alignment pipeline and returns performance metrics."""
        start_time = time.time()
        self.manager.select_sparse_edges(neighbors_per_user=neighbors_per_node)
        self.manager.compute_pairwise_transforms()
        
        if self.use_ceres:
            self.manager.compute_ceres_global_alignment(0, self.gt_poses[0], initialization=lm_init, gt_poses=self.gt_poses)
        elif self.use_gtsam:
            self.manager.compute_gtsam_global_alignment(0, self.gt_poses[0], initialization=lm_init, gt_poses=self.gt_poses)
        elif self.use_lm:
            self.manager.compute_lm_global_alignment(0, self.gt_poses[0], initialization=lm_init, gt_poses=self.gt_poses)
        elif use_irls:
            self.manager.compute_spectral_global_alignment(0, self.gt_poses[0])
        else:
            self.manager.compute_l2_global_alignment(0, self.gt_poses[0])
            
        duration = time.time() - start_time
        rmses = []
        for i in self.manager.user_ids:
            t_calc = self.manager.get_global_transform(i)
            t_gt = self.gt_poses[i]
            err_t = np.linalg.norm(t_calc[:3, 3] - t_gt[:3, 3])
            rmses.append(err_t)

        return {
            "mean_rmse": np.mean(rmses),
            "duration": duration,
            "edges_count": len(self.manager.edge_transforms)
        }

def run_salience_study():
    tester = SpectralStressTester()
    landmark_counts = [500, 200, 100, 50]
    n_users = 60
    print(f"\nSalience Study: {n_users} Users, Grid Topology, 2cm Noise")
    for m in landmark_counts:
        tester.generate_synthetic_data(n_users=n_users, topology='grid', n_landmarks=m, 
                                     outlier_ratio=0.1, noise_std=0.02, max_visibility_dist=6.0)
        res = tester.run_test(neighbors_per_node=3)
        print(f"{m:<10} | {res['mean_rmse']:<10.4f} | {res['duration']:<10.4f} | {res['edges_count']:<10}")

if __name__ == "__main__":
    run_salience_study()
