import gtsam
import numpy as np

def test_constructors():
    print("Testing gtsam.Pose3()...")
    try:
        p1 = gtsam.Pose3()
        print("Success: Default constructor works.")
    except Exception as e:
        print(f"Failed: {e}")

    print("\nTesting gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0,0,0))...")
    # This crashed last time, skipping to try others first or trying to isolate
    
    print("\nTesting gtsam.Pose3.Identity()...")
    try:
        p2 = gtsam.Pose3.Identity()
        print("Success: .Identity() works.")
    except Exception as e:
        print(f"Failed: {e}")

    print("\nTesting gtsam.Pose3(np.eye(4))...")
    try:
        mat = np.eye(4, dtype=np.float64)
        p3 = gtsam.Pose3(mat)
        print("Success: 4x4 Matrix constructor works.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_constructors()
