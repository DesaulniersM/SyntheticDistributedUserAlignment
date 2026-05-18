import numpy as np
try:
    from .SimpleICP import SimpleICP
except ImportError:
    from SimpleICP import SimpleICP

class AlignmentSolver:
    def __init__(self):
        # Stateless
        self.icp_error_reset_threshold = 0.05

    def get_rotation_between_vectors(self, a, b):
        """Computes rotation matrix that maps vector a to align with vector b."""
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        v = np.cross(a, b); c = np.dot(a, b); s = np.linalg.norm(v)
        if s < 1e-8:
            return np.eye(3) if c > 0 else -np.eye(3) # Simple flip for 180
        k_mat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + k_mat + (k_mat @ k_mat) * ((1 - c) / (s**2))

    def run_configured_solver(self, host_points, local_points, host_gravity=None, local_gravity=None, correspondences=None):
        """
        Master solver that performs 4-DoF alignment locked to the Z-up gravity vector.
        """
        host_points = np.array(host_points)
        local_points = np.array(local_points)
        
        # In Z-Up, Gravity points to [0, 0, -1]. 
        # But we align the "UP" vector to [0, 0, 1].
        world_up = np.array([0, 0, 1])

        if host_gravity is not None and local_gravity is not None:
            # Gravity here is actually the LOCAL UP vector of the device
            r_host = self.get_rotation_between_vectors(host_gravity, world_up)
            r_local = self.get_rotation_between_vectors(local_gravity, world_up)
            
            pts_host_leveled = (r_host @ host_points.T).T
            pts_local_leveled = (r_local @ local_points.T).T
            
            # Run Robust 4-DoF ICP in the leveled Z-up space
            t_yaw, solver_error = SimpleICP.solve_robust(
                pts_host_leveled, 
                pts_local_leveled,
                correspondences=correspondences
            )
            
            # Reconstruct the final transform in the original local frame
            t_r_host = np.eye(4); t_r_host[:3, :3] = r_host
            t_r_local = np.eye(4); t_r_local[:3, :3] = r_local
            
            # Relationship: P_host = T_final @ P_local
            # P_host_leveled = T_yaw @ P_local_leveled
            # R_host @ P_host = T_yaw @ (R_local @ P_local)
            # P_host = (R_host^-1 @ T_yaw @ R_local) @ P_local
            final_t = np.linalg.inv(t_r_host) @ t_yaw @ t_r_local
            
            return final_t, solver_error

        # Fallback if no gravity provided (Standard 6-DoF or naive Z-Yaw)
        return SimpleICP.solve_robust(host_points, local_points, correspondences=correspondences)
