
import requests
import time
import numpy as np
import open3d as o3d
from common.conventions import get_robotic_pose
from solvers.global_solvers.MultiUserAlignment import SpectralAlignmentManager

def generate_unified_map():
    print("--- GENERATING UNIFIED GLOBAL MAP (VSLR-SYNC) ---")
    
    nodes = {
        0: "http://localhost:8001",
        1: "http://localhost:8002",
        2: "http://localhost:8003",
        3: "http://localhost:8004",
        4: "http://localhost:8005",
        5: "http://localhost:8006"
    }
    
    # Use the proven Parallel Eye setup
    base_x, base_z = -1.0, 0.5
    look_dist = 2.0
    up = [0, 0, 1]
    
    gt_poses = {}
    for i in range(len(nodes)):
        pos = [base_x, (i - 2.5) * 0.15, base_z] 
        look_at = [base_x + look_dist, pos[1], base_z]
        gt_poses[i] = get_robotic_pose(pos, look_at)

    print("Step 1: Capturing scans from all nodes...")
    for uid, url in nodes.items():
        pos = gt_poses[uid][:3, 3].tolist()
        look_at = [pos[0] + look_dist, pos[1], pos[2]]
        requests.post(f"{url}/scan", json={
            "pos": pos, "look": look_at, "up": up, "use_wavelets": True
        })

    print("Step 2: Performing Peer-to-Peer Alignment...")
    manager = SpectralAlignmentManager()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            try:
                peer_internal_url = f"http://node{j+1}:{8001+j}"
                resp = requests.post(f"{nodes[i]}/align-with-peer", params={
                    "peer_url": peer_internal_url, "use_wavelets": True
                })
                res = resp.json()
                if res["confidence"] > 0.02:
                    manager.edge_transforms[(i, j)] = np.array(res["transform"])
                    manager.edge_weights[(i, j)] = res["confidence"]
            except: pass

    print("Step 3: Synchronizing into Global Frame...")
    for uid, url in nodes.items():
        data = requests.get(f"{url}/points").json()
        manager.add_user_data(uid, np.array(data["points"]))

    manager.compute_spectral_global_alignment(0, gt_poses[0])

    # Merging point clouds for export AND visualization
    geometries = []
    
    # 1. Add Room as Wireframe
    try:
        mesh = o3d.io.read_triangle_mesh("replica_office1.ply")
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        wireframe.paint_uniform_color([0.3, 0.3, 0.3])
        geometries.append({'name': 'Room', 'geometry': wireframe})
    except: pass

    colors = [[0,1,0], [1,0,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]
    merged_cloud = o3d.geometry.PointCloud()
    
    for uid in manager.user_ids:
        color = colors[uid % len(colors)]
        t_global = manager.get_global_transform(uid)
        pts_world = manager.get_global_cloud(uid)
        
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_world))
        pcd.paint_uniform_color(color)
        merged_cloud += pcd
        
        # Add Camera Frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        frame.transform(t_global)
        geometries.append({'name': f'User {uid}', 'geometry': frame})
        geometries.append({'name': f'User {uid} Cloud', 'geometry': pcd})

    # Save for export
    output_file = "results/unified_map.ply"
    o3d.io.write_point_cloud(output_file, merged_cloud)
    print(f"\nSUCCESS: Unified map saved to {output_file}")
    
    print("Opening 3D Visualizer...")
    o3d.visualization.draw(geometries)

if __name__ == "__main__":
    generate_unified_map()
