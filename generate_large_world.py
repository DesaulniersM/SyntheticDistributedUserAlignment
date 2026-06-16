import numpy as np
import os

def generate_large_world():
    # 1. Generate 225 User Poses in 15x15 grid (approx 37m x 37m area)
    # 37m / 14 intervals ~= 2.64m spacing
    x = np.linspace(0, 37, 15)
    y = np.linspace(0, 37, 15)
    xv, yv = np.meshgrid(x, y)
    user_positions = np.stack([xv.ravel(), yv.ravel(), np.zeros(225)], axis=1)
    
    # Fixed seed for reproducibility
    np.random.seed(42)
    
    user_poses = []
    for pos in user_positions:
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T[:3, 3] = pos
        user_poses.append(T)
    
    # 2. Generate 10,000 Landmarks in 100 anchors
    # 10x10 grid of anchors over 37x37m
    ax = np.linspace(0, 37, 10)
    ay = np.linspace(0, 37, 10)
    axv, ayv = np.meshgrid(ax, ay)
    anchors = np.stack([axv.ravel(), ayv.ravel()], axis=1) # 100 anchors
    
    landmarks = []
    points_per_anchor = 100
    for anchor in anchors:
        # 100 points per cluster in a 1m cube horizontal spread
        # Vertical Stratification: 20% Floor (0-0.5m), 60% Interaction (0.8-1.5m), 20% Ceiling (2.0-2.5m)
        for (z_low, z_high), count in [((0, 0.5), 20), ((0.8, 1.5), 60), ((2.0, 2.5), 20)]:
            pts = np.zeros((count, 3))
            pts[:, 0] = anchor[0] + np.random.uniform(-0.5, 0.5, count)
            pts[:, 1] = anchor[1] + np.random.uniform(-0.5, 0.5, count)
            pts[:, 2] = np.random.uniform(z_low, z_high, count)
            landmarks.append(pts)
            
    landmarks = np.vstack(landmarks)
    
    data = {
        'user_poses': np.array(user_poses),
        'landmarks': landmarks
    }
    
    os.makedirs('results', exist_ok=True)
    np.save('results/world_geometry_large.npy', data)
    print(f"Generated large master world with {len(user_poses)} users and {len(landmarks)} landmarks.")

if __name__ == "__main__":
    generate_large_world()
