"""
Test script for verifying Incremental BFS alignment with 5 simulated users.
"""
import requests
import open3d as o3d
import numpy as np
import time
from IncrementalAlignment import GlobalAlignmentManager

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

def run_5_user_test():
    print("--- Starting Incremental Global Alignment Test (5 Users) ---")
    
    look_at = [1.24, 2.85, -1.13]
    up = [0, 1, 0]
    base_pos = [8.24, 11.61, -9.13]
    
    # Generate 5 poses in a circle
    poses = {}
    vec = np.array(base_pos) - np.array(look_at)
    for i in range(5):
        # Rotate by 72 degrees each (360/5)
        pos = np.array(look_at) + rotate_vector(vec, i * 72, axis='y')
        poses[i] = pos.tolist()

    manager = GlobalAlignmentManager()
    view_matrices = {}
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
            print(f"User {uid} data acquired.")
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

    # 3. Incremental Solve (Anchored to User 0 World Pose)
    print("Computing Incremental Synchronization...")
    anchor_world_pose = np.linalg.inv(view_matrices[0])
    manager.compute_incremental_global_alignment(anchor_id=0, anchor_world_pose=anchor_world_pose)

    # Visualization
    visual_geometries = []
    lab_mesh = o3d.io.read_triangle_mesh("labModel.obj")
    lab_mesh.triangle_material_ids = o3d.utility.IntVector()
    lab_mesh.paint_uniform_color([0.8, 0.8, 0.8])
    visual_geometries.append({'name': 'Laboratory', 'geometry': lab_mesh})

    palette = [[0,0.8,0], [0.8,0,0], [0,0.5,1], [1,0.5,0], [0.8,0,0.8]]

    for uid in manager.user_ids:
        color = palette[uid % len(palette)]
        global_points = manager.get_global_cloud(uid)
        if global_points.size == 0: continue
        
        world_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(global_points))
        world_pts.paint_uniform_color(color)
        
        t_global = manager.get_global_transform(uid)
        frustum = create_camera_frustum(np.eye(4), color, scale=2.0)
        frustum.transform(t_global)
        
        visual_geometries.append({'name': f'User {uid} Cloud', 'geometry': world_pts})
        visual_geometries.append({'name': f'User {uid} Cam', 'geometry': frustum})

    print("Opening 3D Visualization...")
    o3d.visualization.draw(visual_geometries, show_ui=True)

if __name__ == "__main__":
    run_5_user_test()
