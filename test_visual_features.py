
import requests
import cv2
import numpy as np
import base64
import os
from common.visual_features import VisualFeatureEngine

def run_visual_test():
    print("--- VISUAL FEATURE (ORB-3D) TEST ---")
    
    # 1. Capture a scan with a rich view
    pos = [0.6, 0.7, 0.8]
    look = [2.0, 1.5, 0.8]
    up = [0, 0, 1]
    
    print(f"Requesting RGB-D scan...")
    resp = requests.post("http://localhost:8000/scan", json={
        "position": pos, "look_at": look, "up": up
    })
    
    data = resp.json()
    
    # Decode Image
    img_bytes = base64.b64decode(data["image_base64"])
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    
    # Decode Depth (16-bit PNG)
    depth_bytes = base64.b64decode(data["depth_base64"])
    depth_arr = np.frombuffer(depth_bytes, dtype=np.uint8)
    depth_img = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)
    depth_map = depth_img.astype(np.float32) / 1000.0 # Back to meters
    
    # 2. Extract 3D Features
    engine = VisualFeatureEngine()
    descriptors, points_3d, gists = engine.extract_3d_features(img, depth_map)
    
    print(f"Detected {len(descriptors)} ORB features with valid 3D coordinates.")
    
    # 3. Draw features on image for verification
    # We'll just draw circles at the keypoint locations
    # (Since we didn't return keypoints directly, let's re-run detection for viz)
    kp = engine.orb.detect(img, None)
    img_viz = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    
    cv2.imwrite("visual_features_test.png", img_viz)
    print("Verification image saved to visual_features_test.png")
    
    # Print sample 3D point
    if len(points_3d) > 0:
        print(f"Sample 3D Feature (Local Robotics Frame): {points_3d[0]}")

if __name__ == "__main__":
    run_visual_test()
