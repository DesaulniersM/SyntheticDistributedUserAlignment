import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, Optional, List

class SimpleICP:
    """
    A specialized Iterative Closest Point (ICP) implementation for 4-DoF alignment.
    Convention: Right-Handed, Z-Up (X-Forward, Y-Left).
    """
    
    SCALE = 1.0

    @staticmethod
    def downsample(points: np.ndarray, target: int = 10000) -> np.ndarray:
        if len(points) > target:
            step = len(points) // target
            return points[::step][:target]
        return points

    @staticmethod
    def find_optimal_transform_4dof(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
        """Recovers Z-Yaw and 3D Translation."""
        if len(source) < 2: return np.eye(4), 0.0
        cs = np.mean(source, axis=0); ct = np.mean(target, axis=0)
        ps = source - cs; pt = target - ct
        
        # 2D Covariance on local XY plane
        H = np.array([
            [np.sum(ps[:, 0] * pt[:, 0]), np.sum(ps[:, 0] * pt[:, 1])],
            [np.sum(ps[:, 1] * pt[:, 0]), np.sum(ps[:, 1] * pt[:, 1])]
        ])
        
        U, S, Vt = np.linalg.svd(H)
        R_2d = Vt.T @ U.T
        if np.linalg.det(R_2d) < 0:
            Vt[1, :] *= -1
            R_2d = Vt.T @ U.T
            
        theta = np.arctan2(R_2d[1, 0], R_2d[0, 0])
        c, s = np.cos(theta), np.sin(theta)
        R_3d = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        t = ct - R_3d @ cs
        T = np.eye(4); T[:3, :3] = R_3d; T[:3, 3] = t
        return T, 0.0

    @staticmethod
    def solve_4dof(source_points: np.ndarray, target_points: np.ndarray, max_iterations: int = 20, initial_transform: Optional[np.ndarray] = None, target_tree: Optional[cKDTree] = None) -> Tuple[np.ndarray, float]:
        src_s = SimpleICP.downsample(source_points, 5000)
        current_t = initial_transform if initial_transform is not None else np.eye(4)
        if target_tree is None:
            tgt_s = SimpleICP.downsample(target_points, 5000)
            target_tree = cKDTree(tgt_s); target_pts_for_idx = tgt_s
        else:
            target_pts_for_idx = target_points

        dist_threshold = 0.5 * SimpleICP.SCALE
        for _ in range(max_iterations):
            src_transformed = (current_t[:3, :3] @ src_s.T).T + current_t[:3, 3]
            distances, indices = target_tree.query(src_transformed)
            mask = distances < dist_threshold
            if np.sum(mask) < 4: break
            delta_t, _ = SimpleICP.find_optimal_transform_4dof(src_s[mask], target_pts_for_idx[indices[mask]])
            current_t = delta_t
            dist_threshold = max(dist_threshold * 0.85, 0.05 * SimpleICP.SCALE)

        final_src = (current_t[:3, :3] @ src_s.T).T + current_t[:3, 3]
        dists, _ = target_tree.query(final_src)
        inlier_mask = dists < (0.1 * SimpleICP.SCALE)
        if np.sum(inlier_mask) < 4: return current_t, float('inf')
        score = np.sqrt(np.mean(dists[inlier_mask]**2))
        return current_t, score

    @staticmethod
    def solve_robust(source_points: np.ndarray, target_points: np.ndarray, ransac_iterations: int = 1000, inlier_threshold: float = 0.1, icp_iterations: int = 30, correspondences: Optional[List[Tuple[int, int]]] = None) -> Tuple[np.ndarray, float]:
        """
        Performs robust 4-DoF alignment using RANSAC followed by ICP.
        If correspondences are provided, RANSAC uses them. Otherwise, it uses random sampling.
        """
        best_t = np.eye(4)
        best_inliers = -1
        
        src_s = SimpleICP.downsample(source_points, 1000)
        tgt_s = SimpleICP.downsample(target_points, 1000)
        tgt_tree = cKDTree(tgt_s)

        if correspondences and len(correspondences) >= 3:
            # RANSAC on known correspondences
            m_pts1 = source_points[[c[0] for c in correspondences]]
            m_pts2 = target_points[[c[1] for c in correspondences]]
            num_corr = len(correspondences)
            
            for _ in range(min(ransac_iterations, 500)):
                idx = np.random.choice(num_corr, 3, replace=False)
                T, _ = SimpleICP.find_optimal_transform_4dof(m_pts1[idx], m_pts2[idx])
                
                # Count inliers on ALL correspondences
                src_transformed = (T[:3, :3] @ m_pts1.T).T + T[:3, 3]
                errs = np.linalg.norm(src_transformed - m_pts2, axis=1)
                inliers = np.sum(errs < inlier_threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_t = T
        else:
            # Standard RANSAC (Blind matching)
            # This is slow and typically only used as a fallback
            for _ in range(ransac_iterations):
                idx_src = np.random.choice(len(src_s), 3, replace=False)
                idx_tgt = np.random.choice(len(tgt_s), 3, replace=False)
                
                T, _ = SimpleICP.find_optimal_transform_4dof(src_s[idx_src], tgt_s[idx_tgt])
                
                # Check inliers against downsampled target cloud
                src_transformed = (T[:3, :3] @ src_s.T).T + T[:3, 3]
                dists, _ = tgt_tree.query(src_transformed)
                inliers = np.sum(dists < inlier_threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_t = T
                    if inliers > (len(src_s) * 0.8): break

        # Refine with ICP
        return SimpleICP.solve_4dof(source_points, target_points, max_iterations=icp_iterations, initial_transform=best_t, target_tree=tgt_tree)
