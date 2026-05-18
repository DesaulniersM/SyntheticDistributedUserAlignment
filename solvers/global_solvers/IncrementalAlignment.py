import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from AlignmentSolver import AlignmentSolver
from scipy.linalg import inv
from collections import deque

class GlobalAlignmentManager:
    """
    Manages global alignment using Path-Based Incremental Synchronization.
    This is highly stable for small groups and avoids the sign sensitivities 
    of the spectral method.
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
        for i in range(n):
            self.edge_transforms[(self.user_ids[i], self.user_ids[(i+1)%n])] = None
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
            # AlignmentSolver handles the gravity locking
            # returns T such that P_tgt = T @ P_src
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

    def compute_incremental_global_alignment(self, anchor_id: int, anchor_world_pose: np.ndarray):
        """
        Propagates transforms from the anchor node using BFS.
        anchor_world_pose is T_anchor_to_world.
        """
        if anchor_id not in self.user_ids: return
        
        # Build adjacency: src -> (tgt, T_src_to_tgt)
        adj = {uid: [] for uid in self.user_ids}
        for (src, tgt), T in self.edge_transforms.items():
            if T is None: continue
            # P_tgt = T @ P_src
            adj[src].append((tgt, T))
            # P_src = inv(T) @ P_tgt
            adj[tgt].append((src, inv(T)))

        queue = deque([anchor_id])
        self.global_transforms = {anchor_id: anchor_world_pose}
        visited = {anchor_id}

        print(f"Propagating from User {anchor_id}...")
        while queue:
            u = queue.popleft()
            T_u_to_world = self.global_transforms[u]
            for v, T_u_to_v in adj[u]:
                if v not in visited:
                    # P_world = T_u_to_world @ P_u
                    # P_v = T_u_to_v @ P_u  =>  P_u = inv(T_u_to_v) @ P_v
                    # P_world = T_u_to_world @ inv(T_u_to_v) @ P_v
                    # So T_v_to_world = T_u_to_world @ inv(T_u_to_v)
                    self.global_transforms[v] = T_u_to_world @ inv(T_u_to_v)
                    visited.add(v)
                    queue.append(v)
                    print(f"  User {v} registered.")

    def get_global_cloud(self, uid: int) -> np.ndarray:
        if uid not in self.user_clouds or uid not in self.global_transforms: return np.array([])
        T = self.global_transforms[uid]; pts = self.user_clouds[uid]
        return (T[:3, :3] @ pts.T).T + T[:3, 3]

    def get_global_transform(self, uid: int) -> np.ndarray:
        return self.global_transforms.get(uid, np.eye(4))
