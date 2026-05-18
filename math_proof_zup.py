import numpy as np
from scipy.linalg import eigh

def get_z_rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def run_proof():
    print("--- 4-DoF Synchronization First-Principles Proof ---")
    
    # 1. Generate 6 Random Poses (Ground Truth)
    # Pose: T_w_i maps local to world
    gt_poses = []
    for i in range(6):
        yaw = np.random.uniform(0, 2*np.pi)
        pos = np.random.uniform(-5, 5, 3)
        T = np.eye(4); T[:3, :3] = get_z_rot(yaw); T[:3, 3] = pos
        gt_poses.append(T)
        
    # 2. Generate Relative Transforms (Perfect Observations)
    # T_ij s.t. P_i = T_ij @ P_j  => T_ij = T_w_i^-1 @ T_w_j
    # This means T_w_j = T_w_i @ T_ij
    edges = []
    for i in range(6):
        for j in range(i + 1, 6):
            T_ij = np.linalg.inv(gt_poses[i]) @ gt_poses[j]
            edges.append((i, j, T_ij))
            
    # 3. Solve Angular Sync
    # theta_j = theta_i + theta_ij => theta_j - theta_i = theta_ij
    # H[j, i] = e^{i(theta_j - theta_i)} = e^{i*theta_ij}
    n = 6
    H = np.zeros((n, n), dtype=complex)
    for i, j, T in edges:
        theta_ij = np.arctan2(T[1, 0], T[0, 0])
        z_ij = np.exp(1j * theta_ij)
        H[j, i] = z_ij
        H[i, j] = np.conj(z_ij)
    for k in range(n): H[k, k] = 1.0
    
    _, vecs = np.linalg.eigh(H); v = vecs[:, -1]
    
    # Anchor to GT Node 0
    yaw_0_gt = np.arctan2(gt_poses[0][1, 0], gt_poses[0][0, 0])
    offset_phasor = np.exp(1j * yaw_0_gt) / v[0]
    recovered_yaws = np.angle(v * offset_phasor)
    
    # Verify Yaws
    yaw_err = 0
    for i in range(n):
        yaw_gt = np.arctan2(gt_poses[i][1, 0], gt_poses[i][0, 0])
        yaw_err += abs(np.arctan2(np.sin(recovered_yaws[i] - yaw_gt), np.cos(recovered_yaws[i] - yaw_gt)))
    print(f"Mean Yaw Error: {yaw_err/n:.10f} rad")
    
    # 4. Solve Translation Sync
    # t_j - t_i = R_i @ t_ij
    L = np.zeros((n, n)); b = np.zeros((n, 3))
    rec_rots = [get_z_rot(y) for y in recovered_yaws]
    for i, j, T in edges:
        rhs = rec_rots[i] @ T[:3, 3]
        L[j, j] += 1; L[i, i] += 1; L[j, i] -= 1; L[i, j] -= 1
        b[j] += rhs; b[i] -= rhs
        
    # Anchor Translation
    L[0, 0] += 1e9; b[0] += 1e9 * gt_poses[0][:3, 3]
    recovered_pos, _, _, _ = np.linalg.lstsq(L, b, rcond=None)
    
    # Verify Final RMSE
    total_rmse = 0
    for i in range(n):
        err = np.linalg.norm(recovered_pos[i] - gt_poses[i][:3, 3])
        total_rmse += err
    print(f"Mean Translation Error: {total_rmse/n:.10f}m")
    
    if (yaw_err/n < 1e-8) and (total_rmse/n < 1e-8):
        print("\n[SUCCESS] FIRST-PRINCIPLES MATH IS PROVEN!")
        print("Final Composition Rule:")
        print("  R_j = R_i @ R_ij")
        print("  t_j = R_i @ t_ij + t_i")
    else:
        print("\n[FAIL] Math remains broken.")

if __name__ == "__main__":
    run_proof()
