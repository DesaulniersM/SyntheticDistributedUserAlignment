import open3d as o3d
import numpy as np
import os
import sys
import cv2

# Standard way to find common
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from common.conventions import get_robotic_pose, world_to_local
except ImportError:
    sys.path.append('/app')
    from common.conventions import get_robotic_pose, world_to_local

class Camera:
    def __init__(self, mesh_path, width=640, height=480, fov=60.0):
        self.width = width; self.height = height; self.fov = fov
        
        # Load mesh for raycasting
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.scene.add_triangles(mesh_t)
        
        # Pre-compute vertex colors and triangle indices for rendering
        self.vertex_colors = np.asarray(self.mesh.vertex_colors)
        self.triangles = np.asarray(self.mesh.triangles)
        
        f = (width / 2) / np.tan(np.radians(fov / 2))
        self.K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])

    def capture(self, position, look_at, up, noise_level=0.0):
        eye = o3d.core.Tensor(position, dtype=o3d.core.Dtype.Float32)
        center = o3d.core.Tensor(look_at, dtype=o3d.core.Dtype.Float32)
        up_vec = o3d.core.Tensor(up, dtype=o3d.core.Dtype.Float32)
        
        rays = self.scene.create_rays_pinhole(
            fov_deg=self.fov, center=center, eye=eye, up=up_vec, 
            width_px=self.width, height_px=self.height
        )
        
        ans = self.scene.cast_rays(rays)
        t_hit = ans['t_hit'].numpy()
        prim_ids = ans['primitive_ids'].numpy()
        normals = ans['primitive_normals'].numpy()
        t_hit[np.isinf(t_hit)] = 0
        
        # --- 1. RENDER INTENSITY IMAGE (With Shading & Color) ---
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        valid_mask = t_hit > 0
        
        if np.any(valid_mask):
            hit_tri_indices = prim_ids[valid_mask]
            tris = self.triangles[hit_tri_indices]
            c0 = self.vertex_colors[tris[:, 0]]
            c1 = self.vertex_colors[tris[:, 1]]
            c2 = self.vertex_colors[tris[:, 2]]
            avg_colors = (c0 + c1 + c2) / 3.0
            
            # Dual-source Lighting (Top-down + Side)
            light1 = np.array([0, 0, 1])
            light2 = np.array([1, 1, 0]) / np.sqrt(2)
            
            hit_normals = normals[valid_mask]
            shading1 = np.abs(np.sum(hit_normals * light1, axis=1))
            shading2 = np.abs(np.sum(hit_normals * light2, axis=1))
            
            # Mix shading (0.2 ambient + 0.4 top + 0.4 side)
            shading = 0.2 + 0.4 * shading1 + 0.4 * shading2
            
            final_colors = (avg_colors * shading[:, np.newaxis] * 255).astype(np.uint8)
            img[valid_mask] = final_colors

        # --- 2. GENERATE POINT CLOUD ---
        v, u = np.where(valid_mask); z = t_hit[v, u]
        rays_np = rays.numpy()
        dirs = rays_np[v, u, 3:6]
        origins = rays_np[v, u, 0:3]
        pts_world = origins + dirs * z[:, np.newaxis]
        
        T_w = get_robotic_pose(position, look_at)
        pts_local = world_to_local(pts_world, T_w)
        
        # Flip vertically to match standard image coordinates (Origin at Top-Left)
        # Open3D raycasting often results in inverted Y-axis relative to OpenCV
        img = cv2.flip(img, 0)
        t_hit = cv2.flip(t_hit, 0)
        
        return img, t_hit, pts_local

if __name__ == "__main__":
    # Better pose: lower and looking at a specific feature-rich wall
    cam = Camera("replica_office1.ply")
    pos = [0.6, 0.7, 0.5] 
    look = [2.0, 1.5, 0.5]
    img, _, points = cam.capture(pos, look, [0, 0, 1])
    if img is not None:
        cv2.imwrite("render_test.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Captured {len(points)} points. Image saved to render_test.png")
