import requests
import numpy as np
import time

def run_test():
    look_at = [1.24, 2.85, -1.13]
    up = [0, 1, 0]
    pos_2 = [8.24, 11.61, -9.13]
    # Small yaw for easier initial success
    yaw_deg = 15
    rad = np.radians(yaw_deg)
    matrix = np.array([
        [ np.cos(rad), 0, np.sin(rad)],
        [ 0,           1, 0          ],
        [-np.sin(rad), 0, np.cos(rad)]
    ])
    vec = np.array(pos_2) - np.array(look_at)
    pos_1 = (np.array(look_at) + matrix @ vec).tolist()

    print(f"Testing {yaw_deg} deg offset...")
    try:
        t_pts = requests.post("http://localhost:8000/scan", json={"position": pos_2, "look_at": look_at, "up": up}).json()["points"]
        s_pts = requests.post("http://localhost:8000/scan", json={"position": pos_1, "look_at": look_at, "up": up}).json()["points"]
        
        requests.post("http://localhost:8002/set_reference", json={"points": t_pts})
        
        start = time.time()
        resp = requests.post("http://localhost:8002/align", json={
            "source_points": s_pts,
            "host_gravity": [0, -1, 0],
            "local_gravity": [0, -1, 0]
        })
        end = time.time()
        
        data = resp.json()
        found_t = np.array(data["transform"])
        found_yaw = np.degrees(np.arctan2(found_t[0, 2], found_t[0, 0]))
        print(f"  Result: Found Yaw {found_yaw:.2f} deg, Error {data['error']:.4f}, Time {end-start:.2f}s")
        return data['error']
    except Exception as e:
        print(f"  Test failed: {e}")
        return 1.0

if __name__ == "__main__":
    run_test()
