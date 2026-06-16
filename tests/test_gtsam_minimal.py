import gtsam
import numpy as np

def test_minimal_gtsam():
    print(">>> Starting GTSAM Minimal Sanity Check")
    
    try:
        # Step 1: Initialize a simple Point3
        # Use explicit float() casting to be safe
        p = gtsam.Point3(float(1.0), float(2.0), float(3.0))
        print(f"1. Point3 Initialized: {p}")
        
        # Step 2: Initialize a simple Rot3 (Identity)
        r = gtsam.Rot3()
        print(f"2. Rot3 (Identity) Initialized: {r.matrix()}")
        
        # Step 3: Initialize Pose3 (Identity)
        pose_id = gtsam.Pose3(r, p)
        print(f"3. Pose3 (Identity + Point) Initialized: {pose_id}")
        
        # Step 4: Initialize from a 4x4 matrix (The risky part)
        # Identity matrix as a starting point
        mat = np.eye(4, dtype=np.float64)
        pose_mat = gtsam.Pose3(mat)
        print("4. Pose3 from Matrix Initialized successfully.")
        
        print("\n>>> SUCCESS: GTSAM is stable in this environment.")
        
    except Exception as e:
        print(f"\n>>> FAILED: {str(e)}")

if __name__ == "__main__":
    test_minimal_gtsam()
