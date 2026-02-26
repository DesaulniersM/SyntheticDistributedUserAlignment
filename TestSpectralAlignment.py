"""
Test script for verifying Spectral Synchronization with 6 simulated users.
"""
import requests
import open3d as o3d
import numpy as np
import time
from MultiUserAlignment import SpectralAlignmentManager

def rotate_vector(vector, angle_degrees, axis='y'):
    rad = np.radians(angle_degrees)
    matrix = np.array([
        [ np.cos(rad), 0, np.sin(rad)],
        [ 0,           1, 0          ],
        [-np.sin(rad), 0, np.cos(rad)]
    ])
    return matrix @ np.array(vector)

def get_view_matrix(position, look_at, up):
    z_axis = np.array(look_at) - np.array(position)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(np.array(up), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    r = np.vstack([x_axis, y_axis, z_axis])
    t = -r @ np.array(position)
    view_mat = np.eye(4)
    view_mat[:3, :3] = r
    view_mat[:3, 3] = t
    return view_mat

def create_camera_frustum(view_matrix, color, scale=1.0):
    pts = np.array([[0,0,0], [-0.5,-0.5,1], [0.5,-0.5,1], [0.5,0.5,1], [-0.5,0.5,1]]) * scale
    lines = [[0,1], [0,2], [0,3], [0,4], [1,2], [2,3], [3,4], [4,1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    ls.transform(np.linalg.inv(view_matrix))
    return ls

def run_spectral_test():
    print("--- Starting Spectral Synchronization Test (6 Users) ---")
    
    look_at = [1.24, 2.85, -1.13]
    up = [0, 1, 0]
    base_pos = [8.24, 11.61, -9.13]
    
    # Generate 6 poses in a circle
    poses = {}
    vec = np.array(base_pos) - np.array(look_at)
    for i in range(6):
        # Rotate by 60 degrees each
        pos = np.array(look_at) + rotate_vector(vec, i * 60, axis='y')
        # Add some moderate jitter (0.2m) to avoid perfect symmetry issues
        pos += np.random.normal(0, 0.2, 3)
        poses[i] = pos.tolist()

    manager = SpectralAlignmentManager()
    view_matrices = {}
    gravities = {}
    
    world_gravity = np.array([0, -1, 0])

    print("Requesting scans...")
    for uid, pos in poses.items():
        try:
            resp = requests.post("http://localhost:8000/scan", json={"position": pos, "look_at": look_at, "up": up})
            points = np.array(resp.json()["points"])
            v_mat = get_view_matrix(pos, look_at, up)
            local_grav = v_mat[:3, :3] @ world_gravity
            manager.add_user_data(uid, points, gravity=local_grav)
            view_matrices[uid] = v_mat
            gravities[uid] = local_grav
            print(f"User {uid} ready.")
        except Exception as e:
            print(f"Scan failed: {e}")
            return

    # 1. Sparse Edge Selection (2 neighbors each)
    manager.select_sparse_edges(neighbors_per_user=2)
    print(f"Selected {len(manager.edge_transforms)} sparse edges.")

    # 2. Pairwise ICP
    start = time.time()
    manager.compute_pairwise_transforms()
    print(f"Pairwise alignments done in {time.time()-start:.2f}s")

    # 3. Spectral Solve (Anchored to User 0 World Pose)
    print("Computing Spectral Synchronization...")
    # T_0_to_world is inv(V0)
    anchor_world_pose = np.linalg.inv(view_matrices[0])
    manager.compute_spectral_global_alignment(anchor_id=0, anchor_world_pose=anchor_world_pose)

    print("\n--- Final Alignment Results (Relative to Lab Mesh) ---")
    # Load lab mesh to use as ground truth for RMS
    mesh = o3d.io.read_triangle_mesh("labModel.obj")
    # Sample points from the mesh to create a "target" for distance checking
    mesh_points = np.asarray(mesh.sample_points_uniformly(number_of_points=10000).points)
    from scipy.spatial import cKDTree
    mesh_tree = cKDTree(mesh_points)

    for uid in manager.user_ids:
        t_global = manager.get_global_transform(uid)
        
        # Calculate RMS error against the MESH in world space
        pts_i_world = manager.get_global_cloud(uid)
        dists, _ = mesh_tree.query(pts_i_world)
        rms = np.sqrt(np.mean(dists**2))
        
        # Also show relative pose to User 0 for your reference
        t_world_to_user0 = view_matrices[0]
        t_rel_to_0 = t_world_to_user0 @ t_global
        yaw = np.degrees(np.arctan2(t_rel_to_0[0, 2], t_rel_to_0[0, 0]))
        
        print(f"User {uid}:")
        print(f"  RMS Error vs Mesh: {rms:.6f}m")
        print(f"  Yaw rel to User 0: {yaw:.2f} deg")
        print(f"  World Pos: {t_global[:3, 3]}")

    # Visualization
    visual_geometries = []
    lab_mesh = o3d.io.read_triangle_mesh("labModel.obj")
    lab_mesh.triangle_material_ids = o3d.utility.IntVector()
    lab_mesh.paint_uniform_color([0.8, 0.8, 0.8])
    visual_geometries.append({'name': 'Laboratory', 'geometry': lab_mesh})

    # Color cycle
    palette = [[0,0.8,0], [0.8,0,0], [0,0.5,1], [1,0.5,0], [0.8,0,0.8], [0,0.8,0.8]]

    for uid in manager.user_ids:
        color = palette[uid % len(palette)]
        global_points = manager.get_global_cloud(uid)
        
        # Already in World Space from manager
        world_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(global_points))
        world_pts.paint_uniform_color(color)
        
        # Frustum in World Space
        t_global = manager.get_global_transform(uid)
        frustum = create_camera_frustum(np.eye(4), color, scale=2.0)
        frustum.transform(t_global)
        
        visual_geometries.append({'name': f'User {uid} Cloud', 'geometry': world_pts})
        visual_geometries.append({'name': f'User {uid} Cam', 'geometry': frustum})

    print("Opening 3D Visualization...")
    o3d.visualization.draw(visual_geometries, show_ui=True)

if __name__ == "__main__":
    run_spectral_test()
