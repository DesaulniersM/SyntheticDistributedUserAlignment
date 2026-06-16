
import requests
import time
import numpy as np
import os
import open3d as o3d
import base64
import cv2
from common.conventions import get_robotic_pose
from solvers.global_solvers.MultiUserAlignment import SpectralAlignmentManager

def run_comparative_benchmark():
    print("--- L2 vs. IRLS GLOBAL SYNC BENCHMARK ---")
    
    nodes = {
        0: "http://localhost:8001",
        1: "http://localhost:8002",
        2: "http://localhost:8003",
        3: "http://localhost:8004",
        4: "http://localhost:8005",
        5: "http://localhost:8006"
    }
    
    # Proven Parallel Eye setup
    base_x, base_z = -1.0, 0.5
    look_dist = 2.0
    up = [0, 0, 1]
    
    gt_poses = {}
    for i in range(len(nodes)):
        pos = [base_x, (i - 2.5) * 0.15, base_z] 
        look_at = [base_x + look_dist, pos[1], base_z]
        gt_poses[i] = get_robotic_pose(pos, look_at)

    print("Step 1: Capturing scans...")
    for uid, url in nodes.items():
        pos = gt_poses[uid][:3, 3].tolist()
        look_at = [pos[0] + look_dist, pos[1], pos[2]]
        requests.post(f"{url}/scan", json={"pos": pos, "look": look_at, "up": up, "use_wavelets": True})

    manager = SpectralAlignmentManager()
    print("Step 2: Performing Peer-to-Peer Alignment...")
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            try:
                peer_internal_url = f"http://node{j+1}:{8001+j}"
                resp = requests.post(f"{nodes[i]}/align-with-peer", params={"peer_url": peer_internal_url, "use_wavelets": True})
                res = resp.json()
                if res["confidence"] > 0.02:
                    manager.edge_transforms[(i, j)] = np.array(res["transform"])
                    manager.edge_weights[(i, j)] = res["confidence"]
            except: pass

    for uid, url in nodes.items():
        data = requests.get(f"{url}/points").json()
        manager.add_user_data(uid, np.array(data["points"]))

    # --- Run L2 ---
    print("\n>>> Running Naive L2 Sync...")
    manager.compute_l2_global_alignment(0, gt_poses[0])
    l2_rmse = calculate_rmse(manager, gt_poses)
    
    # --- Run IRLS ---
    print("\n>>> Running IRLS-HWA Sync...")
    manager.compute_spectral_global_alignment(0, gt_poses[0])
    irls_rmse = calculate_rmse(manager, gt_poses)

    print(f"\n--- Comparative Results ---")
    print(f"Naive L2 Mean RMSE:   {l2_rmse:.4f}m")
    print(f"IRLS-HWA Mean RMSE:   {irls_rmse:.4f}m")
    print(f"Improvement:         {((l2_rmse - irls_rmse) / l2_rmse * 100):.2f}%")

def calculate_rmse(manager, gt_poses):
    total_rmse = 0
    for uid in manager.user_ids:
        t_calc = manager.get_global_transform(uid)
        t_gt = gt_poses[uid]
        pts_loc = manager.user_clouds[uid][:100]
        p_calc = (t_calc[:3, :3] @ pts_loc.T).T + t_calc[:3, 3]
        p_gt = (t_gt[:3, :3] @ pts_loc.T).T + t_gt[:3, 3]
        rmse = np.sqrt(np.mean(np.linalg.norm(p_calc - p_gt, axis=1)**2))
        total_rmse += rmse
    return total_rmse / len(manager.user_ids)

if __name__ == "__main__":
    run_comparative_benchmark()
