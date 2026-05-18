
import requests
import open3d as o3d
import numpy as np
import os
from common.conventions import get_robotic_pose, local_to_world, world_to_local

def run_raw_test():
    print("--- RAW WORLD-SPACE OVERLAY TEST ---")
    
    pos = [1.0, 1.0, 1.0]
    look_at = [0.0, 0.0, 1.0]
    up = [0, 0, 1]
    
    # 1. Capture scan
    print(f"Requesting scan at {pos}...")
    resp = requests.post("http://localhost:8000/scan", json={
        "position": pos, "look_at": look_at, "up": up
    })
    
    # Scanner now returns RAW world-space points
    pts_world_from_scanner = np.array(resp.json()["points"])
    print(f"Captured {len(pts_world_from_scanner)} points directly from scanner world-space.")
    
    # 2. Test Localize/Globalize cycle (Internal Math Check)
    T_w = get_robotic_pose(pos, look_at)
    
    # P_loc = T^-1 @ P_world
    pts_local = world_to_local(pts_world_from_scanner, T_w)
    
    # P_world_recovered = T @ P_loc
    pts_world_rec = local_to_world(pts_local, T_w)
    
    # Visualization
    geometries = []
    
    # THE MESH (The ultimate ground truth)
    mesh = o3d.io.read_triangle_mesh("replica_office1.ply")
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    wireframe.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(wireframe)

    # THE RAW POINTS (Should hit the walls if mesh origin matches scanner)
    pcd_raw = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_world_from_scanner))
    pcd_raw.paint_uniform_color([1, 0, 0]) # Red
    geometries.append(pcd_raw)
    
    # THE RECOVERED POINTS (Should match Red points exactly if T_w math is correct)
    pcd_rec = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_world_rec))
    pcd_rec.paint_uniform_color([0, 1, 0]) # Green
    # Shift them slightly so we can see both if they overlap
    pcd_rec.translate([0.01, 0.01, 0.01])
    geometries.append(pcd_rec)

    print("\nVisualizing...")
    print("RED: Raw World Points from Scanner")
    print("GREEN: Points after World -> Local -> World (Math Check)")
    print("GRAY: The Mesh")
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    run_raw_test()
