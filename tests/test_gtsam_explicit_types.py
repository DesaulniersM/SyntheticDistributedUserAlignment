import gtsam
import numpy as np

def test_explicit_types():
    print("Testing Pose3 with explicit Rot3 and numpy array...")
    try:
        r = gtsam.Rot3()
        # Explicit 3x1 float64 array
        t = np.array([0, 0, 0], dtype=np.float64).reshape(3,1)
        p = gtsam.Pose3(r, t)
        print("Success: Pose3(Rot3, 3x1 ndarray) works.")
    except Exception as e:
        print(f"Failed: {e}")

    print("\nTesting Pose3.Expmap with 6x1 twist...")
    try:
        # 6-element vector: [omega_x, omega_y, omega_z, v_x, v_y, v_z]
        xi = np.zeros((6, 1), dtype=np.float64)
        p_exp = gtsam.Pose3.Expmap(xi)
        print("Success: Pose3.Expmap works.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_explicit_types()
