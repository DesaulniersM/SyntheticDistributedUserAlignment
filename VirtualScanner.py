import open3d as o3d
import numpy as np
import cv2

class VirtualScanner:
    def __init__(self, mesh_path, width=320, height=240, fov=60.0):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene.add_triangles(mesh_t)
        self.width, self.height, self.fov = width, height, fov

    def scan_pose(self, position, look_at, up, return_world_coords=False):
        eye_t = o3d.core.Tensor(position, dtype=o3d.core.Dtype.Float32)
        center_t = o3d.core.Tensor(look_at, dtype=o3d.core.Dtype.Float32)
        up_t = o3d.core.Tensor(up, dtype=o3d.core.Dtype.Float32)

        rays = self.scene.create_rays_pinhole(
            fov_deg=self.fov, center=center_t, eye=eye_t, up=up_t,
            width_px=self.width, height_px=self.height,
        )
        ans = self.scene.cast_rays(rays)
        depth_map = ans['t_hit'].numpy()
        depth_map[np.isinf(depth_map)] = 0.0

        # --- GEOMETRIC EDGE SAMPLING ---
        # Mimic mobile AR (ARCore/ARKit) which prioritizes trackable geometric features
        # and depth discontinuities over flat surfaces.
        
        # 1. Normalize depth map for edge detection
        valid_mask = depth_map > 0
        if not np.any(valid_mask):
            return np.array([])
            
        depth_norm = depth_map.copy()
        depth_norm[valid_mask] = (depth_norm[valid_mask] - np.min(depth_norm[valid_mask])) / (np.max(depth_norm[valid_mask]) - np.min(depth_norm[valid_mask]))
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        
        # 2. Detect Geometric Edges (Depth discontinuities)
        edges = cv2.Canny(depth_uint8, 30, 100)
        
        # 3. Create Sampling Priority
        # We want: 50% edges, 50% random surface points
        edge_v, edge_u = np.where(edges > 0)
        surf_v, surf_u = np.where((depth_map > 0) & (edges == 0))
        
        target_count = 3000
        edge_budget = int(target_count * 0.5)
        surf_budget = target_count - edge_budget
        
        # Sample from edges
        if len(edge_v) > edge_budget:
            idx = np.random.choice(len(edge_v), edge_budget, replace=False)
            v_idx_e, u_idx_e = edge_v[idx], edge_u[idx]
        else:
            v_idx_e, u_idx_e = edge_v, edge_u
            surf_budget += (edge_budget - len(edge_v)) # Shift budget to surfaces if few edges
            
        # Sample from surfaces
        if len(surf_v) > surf_budget:
            idx = np.random.choice(len(surf_v), surf_budget, replace=False)
            v_idx_s, u_idx_s = surf_v[idx], surf_u[idx]
        else:
            v_idx_s, u_idx_s = surf_v, surf_u
            
        # Combine
        v_idx = np.concatenate([v_idx_e, v_idx_s])
        u_idx = np.concatenate([u_idx_e, u_idx_s])
        z_vals = depth_map[v_idx, u_idx]

        rays_np = rays.numpy()
        selected_origins = rays_np[v_idx, u_idx, 0:3]
        selected_dirs = rays_np[v_idx, u_idx, 3:6]
        cloud_world = selected_origins + (selected_dirs * z_vals[:, None])
        
        if return_world_coords:
            return cloud_world

        # Transform to Local
        z_axis = np.array(look_at) - np.array(position)
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(np.array(up), z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        r_world_to_local = np.vstack([x_axis, y_axis, z_axis])
        return (r_world_to_local @ (cloud_world - position).T).T

if __name__ == "__main__":
    scanner = VirtualScanner("labModel.obj")
    pos = [8.24, 11.61, -9.13]; look = [1.24, 2.85, -1.13]; up = [0, 1, 0]
    points = scanner.scan_pose(pos, look, up)
    print(f"Generated dense cloud with {len(points)} points.")
