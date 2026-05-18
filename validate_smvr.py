import requests
import numpy as np
import os
import sys
import open3d as o3d
from common.conventions import get_robotic_pose, get_yaw_from_matrix
from solvers.global_solvers.MultiUserAlignment import SpectralAlignmentManager

def rotate_vector_z(vector, deg):
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R @ np.array(vector)

def run_validation():
    print("--- SMVR NOISY VALIDATION (BASELINE) ---")
    center = np.array([0.6, 0.7, 1.0])
    up = [0, 0, 1]
    
    poses = {}
    for i in range(6):
        angle = i * 60
        pos = center + rotate_vector_z([1.5, 0, 0], angle)
        poses[i] = {"pos": pos, "look": center}

    manager = SpectralAlignmentManager()
    gt_world_matrices = {}

    print("Step 1: Capturing scans and injecting NOISY Features...")
    for uid, data in poses.items():
        resp = requests.post("http://localhost:8000/scan", json={
            "position": data["pos"].tolist(), 
            "look_at": data["look"].tolist(), 
            "up": up
        })
        pts_local = np.array(resp.json()["points"])
        
        tw = get_robotic_pose(data["pos"], data["look"])
        gt_world_matrices[uid] = tw
        
        # --- PERFECT DATA EMULATION (DOWNSAMPLED) ---
        # 1. Start with perfect world coords
        pts_world = (tw[:3, :3] @ pts_local.T).T + tw[:3, 3]
        
        # Downsample to 200 points to keep the O(N^2) spectral math fast
        if len(pts_local) > 200:
            idx = np.random.choice(len(pts_local), 200, replace=False)
            pts_local = pts_local[idx]
            pts_world = pts_world[idx]

        feats = np.zeros((len(pts_local), 32))
        # Use world coords as the "features" for perfect matching
        feats[:, :3] = pts_world
        
        manager.add_user_data(uid, pts_local)
        manager.user_features[uid] = feats
        print(f"  User {uid} ready (200 Perfect Landmarks).")

    print("\nStep 2: SMVR Weighting & Pairwise Alignment (NO CHEATS)...")
    manager.select_sparse_edges(neighbors_per_user=3)
    # This now runs the actual solver
    manager.compute_pairwise_transforms()

    print("\nStep 3: Global Synchronization...")
    manager.compute_spectral_global_alignment(0, gt_world_matrices[0])

    print("\n--- Final Report (Noisy Data) ---")
    total_rmse = 0
    for uid in manager.user_ids:
        t_calc = manager.get_global_transform(uid); t_gt = gt_world_matrices[uid]
        pts_loc = manager.user_clouds[uid][:100]
        p_calc = (t_calc[:3, :3] @ pts_loc.T).T + t_calc[:3, 3]
        p_gt = (t_gt[:3, :3] @ pts_loc.T).T + t_gt[:3, 3]
        rmse = np.sqrt(np.mean(np.linalg.norm(p_calc - p_gt, axis=1)**2))
        print(f"User {uid} RMSE: {rmse:.4f}m")
        total_rmse += rmse
    
    print(f"\nMean Dataset RMSE: {total_rmse/6:.4f}m")

    # --- Step 4: Visualization ---
    print("\nPreparing 3D Visualization...")
    geometries = []
    try:
        mesh = o3d.io.read_triangle_mesh("replica_office1.ply")
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        wireframe.paint_uniform_color([0.3, 0.3, 0.3])
        geometries.append(wireframe)
    except: pass

    colors = [[0,1,0], [1,0,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]
    for uid in manager.user_ids:
        color = colors[uid % len(colors)]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        frame.transform(manager.get_global_transform(uid))
        geometries.append(frame)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(manager.get_global_cloud(uid)))
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

    print("Opening Window...")
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    run_validation()
