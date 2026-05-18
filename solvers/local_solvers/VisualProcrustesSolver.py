import numpy as np
from typing import Tuple, List, Optional
from common.visual_features import VisualFeatureEngine
from .SimpleICP import SimpleICP

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as sp_linalg
import sys
import os
try:
    from config.settings import SIGMA_D, SPECTRAL_INLIER_RATIO, RANSAC_ITERATIONS
except ImportError:
    sys.path.append('/app')
    from config.settings import SIGMA_D, SPECTRAL_INLIER_RATIO, RANSAC_ITERATIONS

class VisualProcrustesSolver:
    """
    A high-speed solver that aligns 3D visual landmarks.
    Implements Stage 2 (Spectral Filtering) to reject visual outliers
    before performing a one-shot Procrustes alignment.
    """
    
    def __init__(self, sigma_d=0.02, k_neighbors=20):
        self.feature_engine = VisualFeatureEngine()
        self.sigma_d = sigma_d
        self.k_neighbors = k_neighbors

    def solve(self, des_src, pts_src, des_tgt, pts_tgt, gists_src=None, gists_tgt=None) -> Tuple[np.ndarray, float]:
        """
        Aligns two sets of visual landmarks.
        Returns: (4x4 Transform, Confidence Score)
        """
        # 1. Propose Matches
        if gists_src is not None and gists_tgt is not None:
            # Match using sparse wavelets
            matches = self.feature_engine.match_features(gists_src, gists_tgt, use_wavelets=True)
        else:
            # Match using standard ORB
            matches = self.feature_engine.match_features(des_src, des_tgt)
            
        if len(matches) < 5:
            return np.eye(4), 0.0

        # Extract proposed pairs
        idx_src = [m.queryIdx for m in matches]
        idx_tgt = [m.trainIdx for m in matches]
        m_pts1 = pts_src[idx_src]
        m_pts2 = pts_tgt[idx_tgt]
        num_m = len(matches)

        # 2. Verify Matches (Stage 2: Sparse Spatial Compatibility)
        # Use k-NN to sparsify the compatibility matrix M
        # Only check compatibility with spatial neighbors in the source cloud
        k = min(self.k_neighbors, num_m - 1)
        tree = cKDTree(m_pts1)
        _, neighbors = tree.query(m_pts1, k=k+1) # k+1 because self is included

        rows = []
        cols = []
        data = []

        for i in range(num_m):
            neighbor_indices = neighbors[i]
            
            # Vectorized compatibility check for all neighbors of i
            p1_i = m_pts1[i]; p1_nb = m_pts1[neighbor_indices]
            p2_i = m_pts2[i]; p2_nb = m_pts2[neighbor_indices]
            
            d1 = np.linalg.norm(p1_i - p1_nb, axis=1)
            d2 = np.linalg.norm(p2_i - p2_nb, axis=1)
            
            scores = np.maximum(0, 1.0 - (np.abs(d1 - d2)**2 / self.sigma_d**2))
            
            # Keep non-zero scores
            valid = scores > 1e-6
            rows.extend([i] * np.sum(valid))
            cols.extend(neighbor_indices[valid].tolist())
            data.extend(scores[valid].tolist())

        # Build Sparse Matrix
        M_sparse = csr_matrix((data, (rows, cols)), shape=(num_m, num_m))

        # Power iteration for principal eigenvector (using sparse matrix)
        v = np.ones((num_m, 1))
        for _ in range(20):
            v_new = M_sparse @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-6: break
            v = v_new / norm
            
        # Edge Confidence (Intercluster Score)
        # s = (v^T M v) / ||v||_1 (normalized by number of entries)
        # For sparse, we use the average support of the principal cluster
        confidence = float(v.T @ M_sparse @ v) / num_m
        
        # 3. Filter and Solve (Robust 4-DoF RANSAC)
        v = v.flatten()
        # Use a more stringent threshold from settings
        inlier_threshold = SPECTRAL_INLIER_RATIO * np.max(v)
        inlier_mask = v > inlier_threshold
        
        if np.sum(inlier_mask) < 4:
            # Fallback to median if cluster is very sparse
            inlier_mask = v > np.median(v)
            if np.sum(inlier_mask) < 3:
                return np.eye(4), 0.0

        # Convert inlier mask back to (src_idx, tgt_idx) correspondences for RANSAC
        current_correspondences = []
        inlier_indices = np.where(inlier_mask)[0]
        for idx in inlier_indices:
            current_correspondences.append((idx_src[idx], idx_tgt[idx]))

        # Use the full robust solver which includes a RANSAC loop over these correspondences
        T, _ = SimpleICP.solve_robust(
            pts_src, 
            pts_tgt,
            correspondences=current_correspondences,
            ransac_iterations=RANSAC_ITERATIONS
        )
        
        return T, confidence
