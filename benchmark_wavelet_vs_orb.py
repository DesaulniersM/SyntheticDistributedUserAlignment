
import requests
import time
import numpy as np
from common.conventions import get_robotic_pose
from solvers.global_solvers.MultiUserAlignment import SpectralAlignmentManager

def run_experiment(use_wavelets: bool):
    label = "WAVELET-GIST" if use_wavelets else "ORB-ONLY"
    print(f"\n>>> Running Experiment: {label}")
    
    nodes = {
        0: "http://localhost:8001", 1: "http://localhost:8002", 2: "http://localhost:8003",
        3: "http://localhost:8004", 4: "http://localhost:8005", 5: "http://localhost:8006"
    }
    center = np.array([0.6, 0.7, 0.8]); up = [0, 0, 1]
    
    gt_poses = {}
    for i in range(len(nodes)):
        angle = np.radians(i * 60)
        pos = center + [0.8 * np.cos(angle), 0.8 * np.sin(angle), 0]
        gt_poses[i] = get_robotic_pose(pos, center)

    # Step 1: Perception
    print("  Step 1: Perception...")
    for uid, url in nodes.items():
        requests.post(f"{url}/scan", json={
            "pos": gt_poses[uid][:3, 3].tolist(), "look": center.tolist(), "up": up,
            "use_wavelets": use_wavelets
        })

    # Step 2: Alignment (Timed)
    print("  Step 2: P2P Alignment...")
    manager = SpectralAlignmentManager()
    start_time = time.time()
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            try:
                peer_url = f"http://node{j+1}:{8001+j}"
                resp = requests.post(f"{nodes[i]}/align-with-peer", 
                                    params={"peer_url": peer_url, "use_wavelets": use_wavelets})
                res = resp.json()
                if res["confidence"] > 0.05:
                    manager.edge_transforms[(i, j)] = np.array(res["transform"])
                    manager.edge_weights[(i, j)] = res["confidence"]
            except: pass
            
    # Step 3: Global Sync
    print("  Step 3: Global Sync...")
    for uid, url in nodes.items():
        resp = requests.get(f"{url}/points", timeout=30)
        manager.add_user_data(uid, np.array(resp.json()["points"]))

    manager.compute_spectral_global_alignment(0, gt_poses[0])
    duration = time.time() - start_time

    # Evaluate RMSE
    total_rmse = 0
    for uid in manager.user_ids:
        t_calc = manager.get_global_transform(uid); t_gt = gt_poses[uid]
        pts_loc = manager.user_clouds[uid][:100]
        p_calc = (t_calc[:3, :3] @ pts_loc.T).T + t_calc[:3, 3]
        p_gt = (t_gt[:3, :3] @ pts_loc.T).T + t_gt[:3, 3]
        total_rmse += np.sqrt(np.mean(np.linalg.norm(p_calc - p_gt, axis=1)**2))
    
    mean_rmse = total_rmse / len(manager.user_ids)
    return duration, mean_rmse

if __name__ == "__main__":
    print("--- WAVELET VS ORB RESEARCH BENCHMARK ---")
    
    # Run Baseline
    t_orb, rmse_orb = run_experiment(use_wavelets=False)
    
    # Run Experiment
    t_wav, rmse_wav = run_experiment(use_wavelets=True)
    
    print("\n" + "="*40)
    print("FINAL COMPARISON RESULTS")
    print("="*40)
    print(f"ORB-ONLY:    Time={t_orb:.2f}s, RMSE={rmse_orb:.4f}m")
    print(f"WAVELET-GIST: Time={t_wav:.2f}s, RMSE={rmse_wav:.4f}m")
    
    speedup = ((t_orb - t_wav) / t_orb) * 100 if t_orb > 0 else 0
    print(f"\nSpeedup: {speedup:.1f}%")
    
    if rmse_wav <= rmse_orb * 1.1:
        print("Conclusion: Wavelets are EFFECTIVE (Speedup with minimal accuracy loss)")
    else:
        print("Conclusion: Wavelets require TUNING (Significant accuracy loss)")
