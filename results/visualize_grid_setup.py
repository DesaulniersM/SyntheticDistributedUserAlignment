import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can find the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stress_test_spectral import SpectralStressTester

def generate_report_visuals():
    # 1. Setup the exact scenario from our best benchmark
    tester = SpectralStressTester(use_ransac=False)
    n_users = 60
    m_points = 150
    
    print(f"Generating geometric visualization for {n_users} users...")
    
    # Generate the data (using same seed/params as our 'Optimal' test)
    tester.generate_synthetic_data(n_users=n_users, topology='grid', n_landmarks=m_points, 
                                 outlier_ratio=0.1, noise_std=0.02, max_visibility_dist=6.0)
    
    # 2. Run the actual solver to get the calculated poses
    stats = tester.run_test(neighbors_per_node=3)
    
    # 3. Create the Visualization
    plt.figure(figsize=(12, 10))
    
    # Plot Users (GT vs Calculated)
    for uid in range(n_users):
        gt_t = tester.gt_poses[uid]
        calc_t = tester.manager.get_global_transform(uid)
        
        gx, gy = gt_t[0, 3], gt_t[1, 3]
        cx, cy = calc_t[0, 3], calc_t[1, 3]
        
        # Plot orientation vector (GT)
        yaw_gt = np.arctan2(gt_t[1, 0], gt_t[0, 0])
        plt.arrow(gx, gy, 0.5*np.cos(yaw_gt), 0.5*np.sin(yaw_gt), 
                  head_width=0.2, color='gray', alpha=0.3, label='GT Pose' if uid==0 else "")
        
        # Plot orientation vector (Calculated)
        yaw_calc = np.arctan2(calc_t[1, 0], calc_t[0, 0])
        plt.arrow(cx, cy, 0.5*np.cos(yaw_calc), 0.5*np.sin(yaw_calc), 
                  head_width=0.2, color='blue', alpha=0.8, label='Spectral Pose' if uid==0 else "")

    # Plot Connections (Edges that survived the spectral filter)
    for (src, tgt) in tester.manager.edge_transforms.keys():
        p1 = tester.gt_poses[src][:2, 3]
        p2 = tester.gt_poses[tgt][:2, 3]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', alpha=0.1, linewidth=0.5, label='Valid Edge' if src==0 and tgt==1 else "")

    # Calculate Global Metrics for the Report
    all_errs = []
    for uid in range(n_users):
        err = np.linalg.norm(tester.gt_poses[uid][:3, 3] - tester.manager.get_global_transform(uid)[:3, 3])
        all_errs.append(err)
    
    max_drift = np.max(all_errs)
    mean_rmse = np.mean(all_errs)

    plt.title(f'Spectral Global Consensus Geometry (N={n_users}, M={m_points})\nMean RMSE: {mean_rmse:.4f}m | Max Drift: {max_drift:.4f}m')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.axis('equal')

    plot_path = "results/grid_geometry_view.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Geometry visualization saved to: {plot_path}")
    
    # Print Global Quality Summary
    print("\n" + "="*30)
    print("GLOBAL ACCURACY REPORT")
    print("="*30)
    print(f"Total Users (Nodes):   {n_users}")
    print(f"Active Edges (Links):  {len(tester.manager.edge_transforms)}")
    print(f"Mean Dataset RMSE:    {mean_rmse:.6f} m")
    print(f"Maximum Node Drift:    {max_drift:.6f} m")
    print(f"Solver Time:           {stats['duration']:.2f} s")
    print("="*30)

if __name__ == "__main__":
    generate_report_visuals()
