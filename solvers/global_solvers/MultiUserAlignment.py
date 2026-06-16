import numpy as np
import open3d as o3d
import os
from typing import Dict, List, Tuple, Optional
from scipy.linalg import eigh
from scipy.spatial import cKDTree
import sys

# Standard package imports
try:
    from common.conventions import get_yaw_from_matrix
    from solvers.local_solvers.AlignmentSolver import AlignmentSolver
    from solvers.local_solvers.SimpleICP import SimpleICP
except ImportError:
    # Handle direct root execution or inside docker
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from common.conventions import get_yaw_from_matrix
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'local_solvers'))
    from AlignmentSolver import AlignmentSolver
    from SimpleICP import SimpleICP

import sys
import os
try:
    from config.settings import GLOBAL_SIGMA_D, IRLS_ITERATIONS, EMA_ALPHA, EMA_BETA
except ImportError:
    sys.path.append('/app')
    from config.settings import GLOBAL_SIGMA_D, IRLS_ITERATIONS, EMA_ALPHA, EMA_BETA

class SpectralAlignmentManager:
    """
    Convention: Robotics standard (X-Forward, Y-Left, Z-Up).
    Implements Stage 2 (Spectral Filtering) and Stage 3 (IRLS-HWA) of Li Fang 2025.
    """

    def __init__(self):
        self.user_ids: List[int] = []
        self.user_clouds: Dict[int, np.ndarray] = {}
        self.user_features: Dict[int, np.ndarray] = {}
        self.user_scan_ids: Dict[int, str] = {}
        self.user_gravities: Dict[int, np.ndarray] = {}
        self.edge_transforms: Dict[Tuple[int, int], np.ndarray] = {}
        self.edge_weights: Dict[Tuple[int, int], float] = {}
        self.global_transforms: Dict[int, np.ndarray] = {}
        
        self.solver = AlignmentSolver()
        self.sigma_d = GLOBAL_SIGMA_D # From settings
        self.irls_iterations = IRLS_ITERATIONS # From settings
        self.ema_alpha = EMA_ALPHA; self.ema_beta = EMA_BETA

    def add_user_data(self, user_id: int, points: np.ndarray, gravity: Optional[np.ndarray] = None, scan_id: Optional[str] = None):
        if user_id not in self.user_ids:
            self.user_ids.append(user_id); self.user_ids.sort()
        self.user_clouds[user_id] = points
        self.user_gravities[user_id] = gravity if gravity is not None else np.array([0, 0, 1])
        if scan_id: self.user_scan_ids[user_id] = scan_id

    def _get_spectral_filtered_matches(self, id1: int, id2: int) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Stage 2: Spatial Compatibility + Spectral Decomposition.
        Finds the principal eigenvector of the compatibility matrix M to prune outliers.
        """
        f1 = self.user_features[id1]; f2 = self.user_features[id2]
        p1 = self.user_clouds[id1]; p2 = self.user_clouds[id2]
        
        # 1. Initial Feature Matching (Mutual Nearest Neighbors)
        tree2 = cKDTree(f2); _, idx12 = tree2.query(f1, k=1)
        tree1 = cKDTree(f1); _, idx21 = tree1.query(f2, k=1)
        mutual_matches = [(i, j) for i, j in enumerate(idx12) if idx21[j] == i]
        
        if len(mutual_matches) < 5: return 0.0, []

        # Cap for performance while building M
        if len(mutual_matches) > 250:
            idx = np.random.choice(len(mutual_matches), 250, replace=False)
            mutual_matches = [mutual_matches[i] for i in idx]
            
        print(f"    Building compatibility matrix for {len(mutual_matches)} matches...")
            
        num_m = len(mutual_matches)
        m_pts1 = p1[[m[0] for m in mutual_matches]]
        m_pts2 = p2[[m[1] for m in mutual_matches]]

        # 2. Build Spatial Compatibility Matrix M (Eq 4)
        M = np.zeros((num_m, num_m))
        for i in range(num_m):
            d1 = np.linalg.norm(m_pts1[i] - m_pts1, axis=1)
            d2 = np.linalg.norm(m_pts2[i] - m_pts2, axis=1)
            # SC = [1 - dist_err^2 / sigma^2]+
            M[i, :] = np.maximum(0, 1.0 - (np.abs(d1 - d2)**2 / self.sigma_d**2))

        # 3. Spectral Decomposition (Power Iteration)
        v = np.ones((num_m, 1))
        for _ in range(30):
            v_new = M @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-6: break
            v = v_new / norm
            
        # 4. Intercluster Score s (Confidence of the entire edge)
        score_s = float(v.T @ M @ v) / num_m
        
        # 5. Pruning: Keep only correspondences that agree with the main cluster
        # Elements of v represent confidence in each match [0, 1]
        v = v.flatten()
        threshold = np.median(v) # Dynamic threshold based on cluster density
        cleaned_matches = [mutual_matches[i] for i in range(num_m) if v[i] >= threshold]
        
        return score_s, cleaned_matches

    def compute_pairwise_transforms(self):
        """Pairs calculate T_ij such that P_i = T_ij @ P_j."""
        total_stage2_iterations = 0
        for (src_id, tgt_id) in list(self.edge_transforms.keys()):
            # Run Stage 2 Spectral Filtering
            conf_score, cleaned_matches = self._get_spectral_filtered_matches(src_id, tgt_id)
            total_stage2_iterations += 30 # Each edge uses 30 power iterations

            if conf_score < 0.05 or len(cleaned_matches) < 4:
                print(f"  Edge {src_id}->{tgt_id} rejected by Spectral Filtering (score={conf_score:.4f})")
                del self.edge_transforms[(src_id, tgt_id)]
                continue

            print(f"  Edge {src_id}->{tgt_id}: Spectral Score={conf_score:.4f}, Inliers={len(cleaned_matches)}")

            # Solve using ONLY the spectrally-verified inliers
            T, error = self.solver.run_configured_solver(
                self.user_clouds[src_id], self.user_clouds[tgt_id],
                host_gravity=self.user_gravities[src_id],
                local_gravity=self.user_gravities[tgt_id],
                correspondences=cleaned_matches
            )

            if error == float('inf'):
                del self.edge_transforms[(src_id, tgt_id)]
            else:
                self.edge_transforms[(src_id, tgt_id)] = T
                # Initial weight is the Spectral Intercluster Score
                self.edge_weights[(src_id, tgt_id)] = conf_score
        return total_stage2_iterations

    def _solve_poses_weighted(self, anchor_id: int, anchor_world_pose: np.ndarray) -> Dict[int, np.ndarray]:
        n = len(self.user_ids); id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}

        # 1. Angular Sync (theta_j - theta_i = theta_ij)
        H = np.zeros((n, n), dtype=complex)
        for (src_id, tgt_id), T in self.edge_transforms.items():
            if T is None: continue
            i_tgt, j_src = id_to_idx[tgt_id], id_to_idx[src_id]
            theta_ij = get_yaw_from_matrix(T)
            z_ij = np.exp(1j * theta_ij); w = self.edge_weights.get((src_id, tgt_id), 1.0)
            H[j_src, i_tgt] = w * z_ij; H[i_tgt, j_src] = w * np.conj(z_ij)

        # Add small regularization to diagonal to ensure connectivity/convergence
        for k in range(n): H[k, k] = 1.0 + 1e-6
        _, vecs = np.linalg.eigh(H); v = vecs[:, -1]

        v_anc = v[id_to_idx[anchor_id]]
        if np.abs(v_anc) < 1e-9: v_anc = 1e-9 # Prevent div by zero
        angles_rel = np.angle(v / (v_anc / np.abs(v_anc)))
        rel_rots = []
        for theta in angles_rel:
            c, s = np.cos(theta), np.sin(theta)
            rel_rots.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))

        # 2. Translation Sync (t_j - t_i = R_i @ t_ij)
        L = np.zeros((n, n)); b = np.zeros((n, 3))
        for (src_id, tgt_id), T in self.edge_transforms.items():
            if T is None: continue
            i_tgt, j_src = id_to_idx[tgt_id], id_to_idx[src_id]
            w = self.edge_weights.get((src_id, tgt_id), 1.0)
            rhs = rel_rots[i_tgt] @ T[:3, 3]
            L[j_src, j_src] += w; L[i_tgt, i_tgt] += w; L[j_src, i_tgt] -= w; L[i_tgt, j_src] -= w
            b[j_src] += w * rhs; b[i_tgt] -= w * rhs

        a_idx = id_to_idx[anchor_id]; L[a_idx, a_idx] += 1e9
        t_rel, _, _, _ = np.linalg.lstsq(L, b, rcond=None)

        res = {}
        for k, uid in enumerate(self.user_ids):
            T_rel = np.eye(4); T_rel[:3, :3] = rel_rots[k]; T_rel[:3, 3] = t_rel[k]
            res[uid] = anchor_world_pose @ T_rel
        return res

    def compute_l2_global_alignment(self, anchor_id: int, anchor_world_pose: np.ndarray):
        """
        Performs a single-pass L2 Least Squares synchronization.
        This ignores the IRLS iterative weights and provides a baseline.
        """
        print("Starting Naive L2 Global Synchronization...")
        self.global_transforms = self._solve_poses_weighted(anchor_id, anchor_world_pose)
        print("L2 Registration Complete.")
        return 1 # Single pass

    def compute_spectral_global_alignment(self, anchor_id: int, anchor_world_pose: np.ndarray):
        """Stage 3: IRLS-HWA (Historical Weighted Average)"""
        edges = list(self.edge_transforms.keys())
        wh_history = {edge: [self.edge_weights.get(edge, 1.0)] for edge in edges}
        vt_ema = {edge: 0.0 for edge in edges}; last_delta = {edge: 0.0 for edge in edges}

        print(f"Starting SMVR Stage 3 (IRLS-HWA) for {self.irls_iterations} iterations...")

        for it in range(self.irls_iterations):
            current_poses = self._solve_poses_weighted(anchor_id, anchor_world_pose)
            for edge in edges:
                src, tgt = edge; T_ij_obs = self.edge_transforms[edge]
                if T_ij_obs is None: continue
                T_ij_fit = np.linalg.inv(current_poses[tgt]) @ current_poses[src]
                yaw_fit = get_yaw_from_matrix(T_ij_fit)
                yaw_obs = get_yaw_from_matrix(T_ij_obs)
                delta_ij = abs(np.arctan2(np.sin(yaw_fit - yaw_obs), np.cos(yaw_fit - yaw_obs)))
                
                # EMA Update
                v_next = (self.ema_alpha * (delta_ij - last_delta[edge]) + self.ema_beta * vt_ema[edge] + self.ema_alpha * delta_ij)
                vt_ema[edge] = v_next; last_delta[edge] = delta_ij
                
                # HWA History Update
                wh_history[edge].append(np.exp(-1.0 * v_next))
                self.edge_weights[edge] = np.mean(wh_history[edge])
            
            if it % 5 == 0: print(f"  Iteration {it}...")

        self.global_transforms = self._solve_poses_weighted(anchor_id, anchor_world_pose)
        print("Registration Complete.")
        return self.irls_iterations

    def select_sparse_edges(self, neighbors_per_user: int = 3):
        # Create sparse candidate graph using centroids from features
        # In our stress test, user_features[:, :3] are world coordinates.
        self.edge_transforms = {}
        n = len(self.user_ids)
        if n <= 1: return

        centroids = []
        for uid in self.user_ids:
            if uid in self.user_features and len(self.user_features[uid]) > 0:
                centroids.append(np.mean(self.user_features[uid][:, :3], axis=0))
            else:
                # Fallback to local (less ideal)
                centroids.append(np.mean(self.user_clouds[uid], axis=0))
        centroids = np.array(centroids)
        
        tree = cKDTree(centroids)
        for i, uid in enumerate(self.user_ids):
            # Query k+1 neighbors because the closest is the node itself
            k = min(neighbors_per_user + 1, n)
            dists, indices = tree.query(centroids[i], k=k)
            
            # Ensure indices is iterable even if k=1
            if k == 1:
                indices = [indices]
            
            for idx in indices:
                neighbor_id = self.user_ids[idx]
                if uid == neighbor_id: continue
                
                # Use a canonical order for edge keys
                edge = tuple(sorted((uid, neighbor_id)))
                self.edge_transforms[edge] = None

    def get_global_cloud(self, uid: int) -> np.ndarray:
        T = self.global_transforms.get(uid, np.eye(4)); pts = self.user_clouds[uid]
        return (T[:3, :3] @ pts.T).T + T[:3, 3]

    def get_global_transform(self, uid: int) -> np.ndarray:
        return self.global_transforms.get(uid, np.eye(4))
