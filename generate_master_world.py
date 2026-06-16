import numpy as np
import os

def generate_master_world():
    # 1. Generate 100 User Poses in 10x10 grid (20m x 20m)
    x = np.linspace(0, 20, 10)
    y = np.linspace(0, 20, 10)
    xv, yv = np.meshgrid(x, y)
    user_positions = np.stack([xv.ravel(), yv.ravel(), np.zeros(100)], axis=1)
    
    # Fixed seed for reproducibility of the master world generation itself
    np.random.seed(42)
    
    user_poses = []
    for pos in user_positions:
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T[:3, 3] = pos
        user_poses.append(T)
    
    # 2. Generate 5000 Landmarks in 50 clusters
    # 50 anchors at 2m intervals. 
    # To cover more of the 20x20 room, let's use a roughly square-ish grid if possible, 
    # but 50 doesn't have a perfect square root. 7x7=49.
    # "at 2m intervals" - let's use 10x5 grid with 2m spacing.
    ax = np.arange(0, 20, 4) # 5 values: 0, 4, 8, 12, 16
    ay = np.arange(0, 20, 2) # 10 values: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
    axv, ayv = np.meshgrid(ax, ay)
    anchors = np.stack([axv.ravel(), ayv.ravel()], axis=1) # 50 anchors
    
    landmarks = []
    for anchor in anchors:
        # 100 points per cluster in a 1m cube horizontal spread
        # Implement 'Vertical Stratification': 20% Floor band (0-0.5m), 60% Interaction band (0.8-1.5m), 20% Ceiling band (2.0-2.5m)
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
    np.save('results/world_geometry.npy', data)
    print(f"Generated master world with {len(user_poses)} users and {len(landmarks)} landmarks.")

if __name__ == "__main__":
    generate_master_world()
