"""
Local test suite for verifying the SimpleICP 4-DoF alignment functionality.

This script generates synthetic point clouds, applies a known transformation,
and verifies that the SimpleICP implementation can recover the ground truth.
"""
import numpy as np
from SimpleICP import SimpleICP
import time

def generate_structured_cloud(num_points: int = 1000) -> np.ndarray:
    """
    Generates an L-shaped room structure point cloud for testing.
    
    Args:
        num_points (int): Total number of points to generate.
        
    Returns:
        np.ndarray: Point cloud of shape (num_points, 3).
    """
    n2 = num_points // 2
    
    # Wall 1: X-Z plane, spanning X (0 to 5)
    wall1 = np.zeros((n2, 3))
    wall1[:, 0] = np.linspace(0, 5, n2)
    wall1[:, 1] = np.random.rand(n2) * 2 # Random height Y
    
    # Wall 2: X-Z plane, spanning Z (0 to 5)
    wall2 = np.zeros((num_points - n2, 3))
    wall2[:, 2] = np.linspace(0, 5, num_points - n2)
    wall2[:, 1] = np.random.rand(num_points - n2) * 2 # Random height Y
    
    return np.vstack([wall1, wall2])

def test_robust_alignment():
    """
    Validates the robust 4-DoF alignment (RANSAC + ICP) against a known transform.
    """
    print("--- Testing Robust Alignment (RANSAC + ICP) ---")
    
    # 1. Generate reference (target) cloud
    target = generate_structured_cloud(1000)
    
    # 2. Define a ground truth LARGE 4-DoF transform
    yaw_deg = 120.0
    yaw_rad = np.radians(yaw_deg) 
    t_true = np.array([2.5, 0.0, 1.0])
    
    # Standard Y-up rotation matrix
    r_true = np.array([
        [ np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [ 0,               1, 0              ],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    # Transform source points such that T * source = target
    # Mathematically: source = R^T * (target - T)
    source = (target - t_true) @ r_true
    
    # Add Gaussian noise to make it realistic (e.g. 2cm std dev)
    source += np.random.normal(0, 0.02, source.shape)
    
    print(f"Goal: Find Yaw {yaw_deg:.1f} deg, Translation {t_true}")

    # 3. Solve using Robust method
    start_time = time.time()
    final_transform, error = SimpleICP.solve_robust(
        source, 
        target, 
        ransac_iterations=5000
    )
    duration = time.time() - start_time

    # 4. Extract results from transformation matrix
    r_found = final_transform[:3, :3]
    t_found = final_transform[:3, 3]
    
    # Extract Yaw from rotation matrix (Y-up convention)
    # r[0,0] = cos(yaw), r[0,2] = sin(yaw)
    yaw_found_rad = np.arctan2(r_found[0, 2], r_found[0, 0])
    yaw_found_deg = np.degrees(yaw_found_rad)

    print(f"Found Yaw: {yaw_found_deg:.2f} deg")
    print(f"Found Translation: {t_found}")
    print(f"RMS Error: {error:.6f}")
    print(f"Total Time: {duration:.4f}s")

    # 5. Assertions with tolerances
    # Note: L-shape symmetry might cause 180-degree ambiguity if the shape were 
    # perfectly symmetric, but random height and point distribution usually prevent this.
    yaw_diff = abs(yaw_found_deg - yaw_deg)
    # Handle wrap-around if necessary (not expected for 120 vs found)
    yaw_diff = min(yaw_diff, 360 - yaw_diff)
    
    assert yaw_diff < 3.0, f"Yaw mismatch: found {yaw_found_deg:.2f}, expected {yaw_deg:.2f}"
    assert np.allclose(t_found, t_true, atol=0.2), f"Translation mismatch: found {t_found}, expected {t_true}"
    
    print("\nSUCCESS: Robust alignment recovered accurately!")

if __name__ == "__main__":
    test_robust_alignment()
