import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from AlignmentSolver import AlignmentSolver
from scipy.linalg import svd, block_diag
from collections import deque

class SpectralAlignmentManager:
    """
    Manages global alignment using 1D Spectral Angular Synchronization.
    Specifically optimized for 4-DoF (Yaw + 3D Translation) where Pitch and Roll
    are locked by gravity.
    """

    def __init__(self):
        self.user_ids: List[int] = []
        self.user_clouds: Dict[int, np.ndarray] = {}
        self.user_gravities: Dict[int, np.ndarray] = {}
        # Stores (src, tgt) -> 4x4 transform T such that P_tgt = T @ P_src
        self.edge_transforms: Dict[Tuple[int, int], np.ndarray] = {}
        self.global_transforms: Dict[int, np.ndarray] = {}
        
        self.solver = AlignmentSolver()
        self.world_down = np.array([0, -1, 0])

    def add_user_data(self, user_id: int, points: np.ndarray, gravity: Optional[np.ndarray] = None):
        if user_id not in self.user_ids:
            self.user_ids.append(user_id)
            self.user_ids.sort()
        self.user_clouds[user_id] = points
        self.user_gravities[user_id] = gravity if gravity is not None else self.world_down

    def select_sparse_edges(self, neighbors_per_user: int = 3):
        self.edge_transforms = {}
        n = len(self.user_ids)
        if n < 2: return
        # Ensure connectivity with a ring
        for i in range(n):
            self.edge_transforms[(self.user_ids[i], self.user_ids[(i+1)%n])] = None
        # Add random shortcuts
        for uid in self.user_ids:
            cur = [t for (s,t) in self.edge_transforms.keys() if s == uid]
            to_add = max(0, neighbors_per_user - len(cur))
            cands = [c for c in self.user_ids if c != uid and c not in cur]
            if cands and to_add > 0:
                for neighbor in random.sample(cands, min(len(cands), to_add)):
                    self.edge_transforms[(uid, neighbor)] = None

    def compute_pairwise_transforms(self):
        """Pairs (j -> i) compute T_ij such that P_i = T_ij @ P_j."""
        for (src_id, tgt_id) in list(self.edge_transforms.keys()):
            print(f"Aligning User {src_id} -> {tgt_id}...")
            transform, error = self.solver.run_configured_solver(
                self.user_clouds[src_id],
                self.user_clouds[tgt_id],
                host_gravity=self.user_gravities[src_id],
                local_gravity=self.user_gravities[tgt_id]
            )
            if error == float('inf'):
                del self.edge_transforms[(src_id, tgt_id)]
            else:
                self.edge_transforms[(src_id, tgt_id)] = transform

    def compute_spectral_global_alignment(self, anchor_id: int, anchor_world_pose: np.ndarray):
        """
        Solves for all poses using 1D Angular Sync for Yaw and Graph Laplacian for Translation.
        """
        n = len(self.user_ids)
        if n == 0: return
        id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        
        # 1. Angular Synchronization for Yaw
        # Build Hermitian matrix H where H_ij = e^(i * theta_ij)
        H = np.zeros((n, n), dtype=complex)
        for (src_id, tgt_id), T in self.edge_transforms.items():
            if T is None: continue
            i, j = id_to_idx[tgt_id], id_to_idx[src_id]
            R_ij = T[:3, :3]
            
            # SimpleICP layout: R_00=cos(theta), R_02=-sin(theta)
            c = R_ij[0, 0]
            s_neg = R_ij[0, 2] # This is -sin(theta)
            # z = cos + i*sin
            z_ij = complex(c, -s_neg)
            
            # H_ij = z_ij means we solve for z_j = z_i * z_ij
            H[i, j] = z_ij
            H[j, i] = np.conj(z_ij)
            
        for i in range(n):
            H[i, i] = 1.0

        vals, vecs = np.linalg.eigh(H)
        v = vecs[:, -1] 
        
        global_rots = []
        for i in range(n):
            theta = np.angle(v[i])
            c, s = np.cos(theta), np.sin(theta)
            R = np.eye(3)
            # Reconstruct using SimpleICP convention: R_02 is -sin
            R[0, 0] = c; R[0, 2] = -s
            R[2, 0] = s; R[2, 2] = c
            global_rots.append(R)
            
        # 2. Solve for Translations (L t = b)
        # Relationship: t_j - t_i = -R_i @ t_ij  (Derived from P_i = R_ij P_j + t_ij)
        L = np.zeros((n, n))
        b = np.zeros((n, 3))
        for (src_id, tgt_id), T in self.edge_transforms.items():
            if T is None: continue
            i, j = id_to_idx[tgt_id], id_to_idx[src_id]
            t_ij = T[:3, 3]
            
            # Constraint: t_j - t_i = -R_i @ t_ij
            L[i, i] += 1; L[j, j] += 1
            L[i, j] -= 1; L[j, i] -= 1
            
            rhs = -global_rots[i] @ t_ij
            b[i] -= rhs
            b[j] += rhs

        t_global_raw, _, _, _ = np.linalg.lstsq(L, b, rcond=None)
        
        # 3. Assemble and Re-Anchor
        raw_transforms = {}
        for i, uid in enumerate(self.user_ids):
            T = np.eye(4)
            T[:3, :3] = global_rots[i]
            T[:3, 3] = t_global_raw[i]
            raw_transforms[uid] = T
            
        # Re-anchor to anchor_id world pose
        T_anchor_raw = raw_transforms[anchor_id]
        T_anchor_raw_inv = np.eye(4)
        T_anchor_raw_inv[:3, :3] = T_anchor_raw[:3, :3].T
        T_anchor_raw_inv[:3, 3] = -T_anchor_raw_inv[:3, :3] @ T_anchor_raw[:3, 3]
        
        for uid in self.user_ids:
            self.global_transforms[uid] = anchor_world_pose @ T_anchor_raw_inv @ raw_transforms[uid]

    def get_global_cloud(self, uid: int) -> np.ndarray:
        if uid not in self.user_clouds or uid not in self.global_transforms: return np.array([])
        T = self.global_transforms[uid]; pts = self.user_clouds[uid]
        return (T[:3, :3] @ pts.T).T + T[:3, 3]

    def get_global_transform(self, uid: int) -> np.ndarray:
        return self.global_transforms.get(uid, np.eye(4))
