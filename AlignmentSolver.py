import numpy as np
from SimpleICP import SimpleICP

class AlignmentSolver:
    def __init__(self):
        # Stateless
        self.icp_error_reset_threshold = 0.05

    def get_rotation_between_vectors(self, a, b):
        """Computes rotation matrix that maps vector a to align with vector b."""
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        
        # Cross product gives the axis of rotation
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        
        if s < 1e-8:
            # Vectors are already aligned or opposite
            if c > 0:
                return np.eye(3)
            else:
                # 180 degree rotation around any orthogonal axis
                # Find an orthogonal axis
                ortho = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
                axis = np.cross(a, ortho)
                axis /= np.linalg.norm(axis)
                # Rodrigues formula for 180 deg
                k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                return np.eye(3) + 2 * (k @ k)
                
        k_mat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + k_mat + (k_mat @ k_mat) * ((1 - c) / (s**2))

    def run_configured_solver(self, host_points, local_points, host_gravity=None, local_gravity=None):
        """
        Master solver that performs 4-DoF alignment locked to the gravity vector.
        """
        host_points = np.array(host_points)
        local_points = np.array(local_points)
        
        world_down = np.array([0, -1, 0])

        if host_gravity is not None and local_gravity is not None:
            # 1. Normalize both clouds to a "Y-is-Gravity" frame
            # This ensures SimpleICP's Y-axis rotation is actually Yaw around Gravity.
            
            # R_host: Maps host points to a frame where host_gravity is [0, -1, 0]
            r_host = self.get_rotation_between_vectors(host_gravity, world_down)
            
            # R_local: Maps local points to a frame where local_gravity is [0, -1, 0]
            r_local = self.get_rotation_between_vectors(local_gravity, world_down)
            
            # Transform both clouds
            pts_host_normalized = (r_host @ host_points.T).T
            pts_local_normalized = (r_local @ local_points.T).T
            
            # 2. Run Robust 4-DoF ICP in the gravity-normalized space
            # Find T_yaw such that: pts_local_norm = T_yaw @ pts_host_norm
            t_yaw, solver_error = SimpleICP.solve_robust(pts_host_normalized, pts_local_normalized)
            
            # 3. Reconstruct the final transform in the original local frame
            # Path: Host -> (R_host) -> Normalized -> (T_yaw) -> Normalized -> (inv(R_local)) -> Local
            # T_final = inv(R_local) @ T_yaw @ R_host
            
            # Convert r_host and r_local to 4x4
            t_r_host = np.eye(4); t_r_host[:3, :3] = r_host
            t_r_local_inv = np.eye(4); t_r_local_inv[:3, :3] = np.linalg.inv(r_local)
            
            final_t = t_r_local_inv @ t_yaw @ t_r_host
            
            return final_t, solver_error

        # Fallback if no gravity provided
        return SimpleICP.solve_robust(host_points, local_points)
