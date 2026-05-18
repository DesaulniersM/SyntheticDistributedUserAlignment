import numpy as np
import os
from typing import Dict

def compute_errors(gt_pose: np.ndarray, pred_pose: np.ndarray):
    """
    Computes rotation and translation errors.
    Metric from the paper:
    re = arccos((tr(R_pred^T * R_gt) - 1) / 2)
    te = ||t_pred - t_gt||_2
    """
    R_gt = gt_pose[:3, :3]
    t_gt = gt_pose[:3, 3]
    
    R_pred = pred_pose[:3, :3]
    t_pred = pred_pose[:3, 3]
    
    # Rotation error
    R_diff = R_pred.T @ R_gt
    trace = np.trace(R_diff)
    # Clip trace to handle numerical errors outside [-1, 1]
    re = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    
    # Translation error
    te = np.linalg.norm(t_pred - t_gt)
    
    return re, te

def run_evaluation(ground_truth_poses: Dict[int, np.ndarray], predicted_poses: Dict[int, np.ndarray]):
    """
    Evaluates global alignment accuracy.
    """
    print("\n" + "="*40)
    print("   ALIGNMENT VALIDATION REPORT")
    print("="*40)
    
    total_re = 0
    total_te = 0
    count = 0
    
    # We evaluate relative to the anchor (User 0 usually)
    # If the global system is anchored to User 0, then User 0 error should be 0.
    
    for uid in sorted(ground_truth_poses.keys()):
        if uid not in predicted_poses:
            print(f"User {uid}: [MISSING]")
            continue
            
        re, te = compute_errors(ground_truth_poses[uid], predicted_poses[uid])
        
        print(f"User {uid}:")
        print(f"  Rotation Error:    {re:8.4f} deg")
        print(f"  Translation Error: {te:8.4f} m")
        
        total_re += re
        total_te += te
        count += 1
        
    if count > 0:
        print("-" * 40)
        print(f"MEAN Rotation Error:    {total_re/count:8.4f} deg")
        print(f"MEAN Translation Error: {total_te/count:8.4f} m")
    
    print("="*40 + "\n")

if __name__ == "__main__":
    # Example usage for manual test
    # (Identity vs Identity)
    gt = {0: np.eye(4)}
    pred = {0: np.eye(4)}
    run_evaluation(gt, pred)
