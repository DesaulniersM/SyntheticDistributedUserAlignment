
import requests
import time
import numpy as np
import os
import open3d as o3d
import base64
import cv2
from common.conventions import get_robotic_pose
from solvers.global_solvers.MultiUserAlignment import SpectralAlignmentManager

def run_networked_benchmark():
    print("--- HIGH-SPEED VISUAL SPECTRAL BENCHMARK (DOCKER) ---")
    
    nodes = {
        0: "http://localhost:8001",
        1: "http://localhost:8002",
        2: "http://localhost:8003",
        3: "http://localhost:8004",
        4: "http://localhost:8005",
        5: "http://localhost:8006"
    }
    
    # Parallel Gaze: Nodes in a line along Y, all looking in the +X direction
    base_x, base_z = -1.0, 0.5
    look_dist = 2.0
    up = [0, 0, 1]
    
    gt_poses = {}
    fwd_vectors = []
    
    print("Step 0: Verifying parallel camera orientation...")
    for i in range(len(nodes)):
        pos = [base_x, (i - 2.5) * 0.15, base_z] # 15cm spacing along Y
        look_at = [base_x + look_dist, pos[1], base_z]
        gt_poses[i] = get_robotic_pose(pos, look_at)
        
        # Extract forward vector (X-axis of local frame)
        fwd = gt_poses[i][:3, 0]
        fwd_vectors.append(fwd)
        
    # Verify dot products
    for i in range(1, len(fwd_vectors)):
        dot = np.dot(fwd_vectors[0], fwd_vectors[i])
        print(f"  Node 0 dot Node {i} forward: {dot:.4f}")

    print("\nStep 1: Nodes scanning and extracting WAVELET landmarks locally...")
    start_time = time.time()
    for uid, url in nodes.items():
        pos = gt_poses[uid][:3, 3].tolist()
        # All look in the same +X direction relative to their position
        look_at = [pos[0] + look_dist, pos[1], pos[2]]
        try:
            resp = requests.post(f"{url}/scan", json={
                "pos": pos, "look": look_at, "up": up, "use_wavelets": True
            })
            res = resp.json()
            print(f"  Node {uid}: {res.get('landmarks', 0)} landmarks extracted.")
            
            # --- Save the View with ORB Overlays ---
            img_data = base64.b64decode(res["image_base64"])
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Redetect locally just for visualization
            orb = cv2.ORB_create(nfeatures=1000)
            kp = orb.detect(img, None)
            img_viz = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
            
            fname = f"results/node_{uid}_view.png"
            cv2.imwrite(fname, img_viz)
            print(f"    Saved view to {fname}")
            
        except Exception as e:
            print(f"  Node {uid} failed: {e}")
            return

    print("\nStep 2: Performing high-speed landmark registration...")
    manager = SpectralAlignmentManager()
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            try:
                # Inside docker, nodes use service names
                peer_internal_url = f"http://node{j+1}:{8001+j}"
                resp = requests.post(f"{nodes[i]}/align-with-peer", params={
                    "peer_url": peer_internal_url,
                    "use_wavelets": True
                })
                res = resp.json()
                
                T = np.array(res["transform"])
                conf = res["confidence"]
                
                if conf > 0.05: # Lower threshold for visual features
                    print(f"  Edge {i}<->{j}: Conf={conf:.4f}")
                    manager.edge_transforms[(i, j)] = T
                    manager.edge_weights[(i, j)] = conf
            except Exception as e:
                pass

    print("\nStep 3: Global Spectral Synchronization...")
    for uid, url in nodes.items():
        try:
            # Increase timeout for 250k point transfer
            resp = requests.get(f"{url}/points", timeout=30)
            data = resp.json()
            pts = np.array(data.get("points", []))
            if len(pts) > 0:
                print(f"  Adding User {uid} data ({len(pts)} points)...")
                manager.add_user_data(uid, pts)
        except Exception as e:
            print(f"  Failed to get points from Node {uid}: {e}")

    if not manager.user_ids:
        print("Error: No nodes have data for synchronization.")
        return

    manager.compute_spectral_global_alignment(0, gt_poses[0])
    duration = time.time() - start_time

    print(f"\n--- Final Benchmark Results ---")
    print(f"Total Networked Processing Time: {duration:.2f}s")
    total_rmse = 0
    for uid in manager.user_ids:
        t_calc = manager.get_global_transform(uid)
        t_gt = gt_poses[uid]
        pts_loc = manager.user_clouds[uid][:100]
        p_calc = (t_calc[:3, :3] @ pts_loc.T).T + t_calc[:3, 3]
        p_gt = (t_gt[:3, :3] @ pts_loc.T).T + t_gt[:3, 3]
        rmse = np.sqrt(np.mean(np.linalg.norm(p_calc - p_gt, axis=1)**2))
        print(f"Node {uid} RMSE: {rmse:.4f}m")
        total_rmse += rmse
    
    mean_err = total_rmse / len(manager.user_ids)
    print(f"Mean Dataset RMSE: {mean_err:.4f}m")

    # --- Step 4: Visualization ---
    print("\nPreparing 3D Visualization...")
    geometries = []
    
    # 1. Add Room as Wireframe
    try:
        mesh = o3d.io.read_triangle_mesh("replica_office1.ply")
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        wireframe.paint_uniform_color([0.3, 0.3, 0.3])
        geometries.append(wireframe)
    except: pass

    # 2. Add Users (Clouds and coordinate axes)
    colors = [[0,1,0], [1,0,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]
    for uid in manager.user_ids:
        color = colors[uid % len(colors)]
        t_global = manager.get_global_transform(uid)
        
        # Camera Frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        frame.transform(t_global)
        geometries.append(frame)
        
        # Point Cloud (Recovered)
        pts_world = manager.get_global_cloud(uid)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_world))
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

    print("Opening Open3D Window...")
    o3d.visualization.draw_geometries(geometries, window_name="High-Speed Visual-Spectral Benchmark")

if __name__ == "__main__":
    run_networked_benchmark()
