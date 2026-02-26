import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, Optional

class SimpleICP:
    """
    A specialized Iterative Closest Point (ICP) implementation for 4-DoF alignment.
    
    This class provides methods to align two point clouds by finding the optimal 
    rotation around the Y-axis (Yaw) and 3D translation. It includes both a 
    standard ICP refinement and a robust RANSAC-based global alignment.
    
    Attributes:
        SCALE (float): A multiplier for thresholds to account for mesh scale differences.
                       Default is 6.0 (assuming the original project used meters and 
                       the current mesh is 6x larger).
    """
    
    SCALE = 6.0

    @staticmethod
    def find_optimal_transform_4dof(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculates the optimal 4-DoF transformation (Yaw + Translation) using SVD.
        
        Args:
            source (np.ndarray): Source points, shape (N, 3).
            target (np.ndarray): Corresponding target points, shape (N, 3).
            
        Returns:
            Tuple[np.ndarray, float]: (4x4 transformation matrix, error score).
                                     Error score is currently returned as 0.0.
        """
        if len(source) < 2:
            return np.eye(4), 0.0
            
        # Compute centroids
        cs = np.mean(source, axis=0)
        ct = np.mean(target, axis=0)
        
        # Center the points
        ps = source - cs
        pt = target - ct
        
        # Build 2D Covariance Matrix (X, Z plane for 4-DoF)
        # ps[:, 0] is X, ps[:, 2] is Z
        h_2d = np.array([
            [np.sum(ps[:, 0] * pt[:, 0]), np.sum(ps[:, 0] * pt[:, 2])],
            [np.sum(ps[:, 2] * pt[:, 0]), np.sum(ps[:, 2] * pt[:, 2])]
        ])
        
        # SVD for 2D rotation
        u, s, vh = np.linalg.svd(h_2d)
        r_2d = vh.T @ u.T
        
        # Reflection check for 2D rotation
        if np.linalg.det(r_2d) < 0:
            vh[1, :] *= -1
            r_2d = vh.T @ u.T
            
        # Reconstruct 3D rotation matrix (rotation only around Y axis)
        r_3d = np.eye(3)
        r_3d[0, 0] = r_2d[0, 0]
        r_3d[0, 2] = r_2d[0, 1]
        r_3d[2, 0] = r_2d[1, 0]
        r_3d[2, 2] = r_2d[1, 1]
        
        # Compute translation: t = ct - R * cs
        t = ct - r_3d @ cs
        
        # Assemble 4x4 transform
        transform = np.eye(4)
        transform[:3, :3] = r_3d
        transform[:3, 3] = t
        
        return transform, 0.0

    @staticmethod
    def solve_4dof(
        source_points: np.ndarray, 
        target_points: np.ndarray, 
        max_iterations: int = 10, 
        initial_transform: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Refines alignment using an iterative process (ICP) with a fixed distance threshold.
        
        Args:
            source_points (np.ndarray): Source point cloud (N, 3).
            target_points (np.ndarray): Target point cloud (M, 3).
            max_iterations (int): Maximum number of ICP iterations.
            initial_transform (np.ndarray, optional): Starting 4x4 transformation matrix.
            
        Returns:
            Tuple[np.ndarray, float]: (Final 4x4 transformation matrix, error score).
        """
        current_t = initial_transform if initial_transform is not None else np.eye(4)
        target_tree = cKDTree(target_points)
        
        # Rejection threshold: 10cm scaled
        rejection_dist_sqr = (0.1 * SimpleICP.SCALE)**2
        
        for _ in range(max_iterations):
            # Transform source points with current estimate
            pts_transformed = (current_t[:3, :3] @ source_points.T).T + current_t[:3, 3]
            
            # Find nearest neighbors
            dists, indices = target_tree.query(pts_transformed)
            
            # Filter matches by distance threshold
            mask = dists**2 < rejection_dist_sqr
            if np.sum(mask) < 5:
                break
            
            # Compute best incremental transform for the inliers
            delta, _ = SimpleICP.find_optimal_transform_4dof(
                source_points[mask], 
                target_points[indices[mask]]
            )
            current_t = delta
            
        # Calculate final RMS error for inliers
        pts_final = (current_t[:3, :3] @ source_points.T).T + current_t[:3, 3]
        dists, _ = target_tree.query(pts_final)
        inlier_mask = dists**2 < rejection_dist_sqr
        
        if np.any(inlier_mask):
            rms_error = np.sqrt(np.mean(dists[inlier_mask]**2))
        else:
            rms_error = float('inf')
            
        return current_t, rms_error

    @staticmethod
    def solve_ransac_4dof_stick(
        source_points: np.ndarray, 
        target_points: np.ndarray, 
        ransac_iterations: int = 10000, 
        inlier_threshold: float = 0.05
    ) -> Tuple[np.ndarray, float]:
        """
        Finds global alignment using a RANSAC approach based on point-pair "sticks".
        
        This method picks two random points in the source and tries to find a matching 
        pair in the target with a similar distance between them.
        
        Args:
            source_points (np.ndarray): Source point cloud (N, 3).
            target_points (np.ndarray): Target point cloud (M, 3).
            ransac_iterations (int): Number of RANSAC trials.
            inlier_threshold (float): Distance threshold for inliers (unscaled).
            
        Returns:
            Tuple[np.ndarray, float]: (Best 4x4 transformation matrix, error score).
                                     Returns inf error if no valid model is found.
        """
        num_src, num_tgt = len(source_points), len(target_points)
        if num_src < 2 or num_tgt < 2:
            return np.eye(4), float('inf')

        # Scale thresholds
        scaled_inlier_thresh_sqr = (inlier_threshold * SimpleICP.SCALE)**2
        dist_tolerance = 0.01 * SimpleICP.SCALE
        # Use a moderate stick (0.5m) to balance ambiguity and recall
        min_stick_length = 0.5 * SimpleICP.SCALE
        
        best_transform = np.eye(4)
        best_inlier_count = -1
        
        target_tree = cKDTree(target_points)

        for _ in range(ransac_iterations):
            # 1. Pick a random pair in Source (The "Stick")
            s1_idx, s2_idx = random.sample(range(num_src), 2)
            source_dist = np.linalg.norm(source_points[s1_idx] - source_points[s2_idx])
            
            if source_dist < min_stick_length:
                continue 

            # 2. Try to find a matching "Stick" in Target
            for _ in range(20):
                t1_idx, t2_idx = random.sample(range(num_tgt), 2)
                target_dist = np.linalg.norm(target_points[t1_idx] - target_points[t2_idx])
                
                if abs(source_dist - target_dist) < dist_tolerance:
                    # Check both possible orientations (t1-t2 and t2-t1)
                    for p1, p2 in [(t1_idx, t2_idx), (t2_idx, t1_idx)]:
                        candidate_t, _ = SimpleICP.find_optimal_transform_4dof(
                            source_points[[s1_idx, s2_idx]], 
                            target_points[[p1, p2]]
                        )
                        
                        # Performance optimization: Quick score with sample points
                        sample_count = 30
                        test_indices = random.sample(range(num_src), min(sample_count, num_src))
                        test_pts = (candidate_t[:3, :3] @ source_points[test_indices].T).T + candidate_t[:3, 3]
                        dists, _ = target_tree.query(test_pts)
                        quick_score = np.sum(dists**2 < scaled_inlier_thresh_sqr)

                        # If sample looks promising (> 50% inliers), perform full check
                        if quick_score > (sample_count / 2):
                            all_pts_transformed = (candidate_t[:3, :3] @ source_points.T).T + candidate_t[:3, 3]
                            all_dists, _ = target_tree.query(all_pts_transformed)
                            current_inlier_count = np.sum(all_dists**2 < scaled_inlier_thresh_sqr)

                            if current_inlier_count > best_inlier_count:
                                best_inlier_count = current_inlier_count
                                best_transform = candidate_t
                    break
        
        # Overlap threshold lowered to 15% for disparate views
        if best_inlier_count > (num_src * 0.15):
            return best_transform, 0.0
            
        return np.eye(4), float('inf')

    @staticmethod
    def solve_robust(
        source_points: np.ndarray, 
        target_points: np.ndarray,
        ransac_iterations: int = 10000,
        inlier_threshold: float = 0.05,
        icp_iterations: int = 10
    ) -> Tuple[np.ndarray, float]:
        """
        Combines RANSAC global alignment with ICP refinement.
        
        Args:
            source_points (np.ndarray): Source point cloud.
            target_points (np.ndarray): Target point cloud.
            ransac_iterations (int): Iterations for the RANSAC stage.
            inlier_threshold (float): Inlier threshold for RANSAC and ICP.
            icp_iterations (int): Iterations for the ICP refinement stage.
            
        Returns:
            Tuple[np.ndarray, float]: (Final refined 4x4 transformation matrix, error).
        """
        # Step 1: Global alignment with RANSAC
        ransac_t, error = SimpleICP.solve_ransac_4dof_stick(
            source_points, 
            target_points, 
            ransac_iterations=ransac_iterations,
            inlier_threshold=inlier_threshold
        )
        
        # Step 2: Local refinement with ICP
        if error != float('inf'):
            return SimpleICP.solve_4dof(
                source_points, 
                target_points, 
                max_iterations=icp_iterations,
                initial_transform=ransac_t
            )
        
        return ransac_t, error
