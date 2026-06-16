import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

def visualize_world(file_name='results/world_geometry_large.npy', output_name='results/master_world_large_viz.png'):
    # 1. Load data
    if not os.path.exists(file_name):
        print(f"Error: {file_name} not found.")
        return
    
    data = np.load(file_name, allow_pickle=True).item()
    user_poses = data['user_poses']
    landmarks = data['landmarks']
    
    # Extract user positions
    user_positions = user_poses[:, :3, 3]
    
    # Calculate limits based on data
    max_x = np.max(user_positions[:, 0]) + 2
    max_y = np.max(user_positions[:, 1]) + 2
    
    # 2. Create Figure
    fig = plt.figure(figsize=(18, 6), dpi=300)
    fig.suptitle(f'Master World Geometry: {os.path.basename(file_name)}', fontsize=16)
    
    # Panel 1: 3D Perspective View
    ax1 = fig.add_subplot(131, projection='3d')
    sc1 = ax1.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], 
                      c=landmarks[:, 2], cmap='viridis', s=0.5, alpha=0.3, label='Landmarks')
    ax1.scatter(user_positions[:, 0], user_positions[:, 1], user_positions[:, 2], 
                c='red', marker='o', s=5, label='User Positions')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Perspective View')
    ax1.set_xlim(0, max_x)
    ax1.set_ylim(0, max_y)
    ax1.set_zlim(0, 3.0)
    
    # Panel 2: Top-Down (XY) View
    ax2 = fig.add_subplot(132)
    ax2.scatter(landmarks[:, 0], landmarks[:, 1], 
                      c=landmarks[:, 2], cmap='viridis', s=0.5, alpha=0.3)
    ax2.scatter(user_positions[:, 0], user_positions[:, 1], 
                c='red', marker='o', s=5, label='User Positions (N=225)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down (XY) View')
    ax2.set_xlim(0, max_x)
    ax2.set_ylim(0, max_y)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', markerscale=5)
    
    # Panel 3: Side (XZ) View
    ax3 = fig.add_subplot(133)
    sc3 = ax3.scatter(landmarks[:, 0], landmarks[:, 2], 
                      c=landmarks[:, 2], cmap='viridis', s=0.5, alpha=0.3)
    ax3.scatter(user_positions[:, 0], user_positions[:, 2], 
                c='red', marker='o', s=5)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side (XZ) View - Vertical Stratification')
    ax3.set_xlim(0, max_x)
    ax3.set_ylim(0, 3.0)
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # Band lines
    for h in [0.5, 0.8, 1.5, 2.0]:
        ax3.axhline(y=h, color='gray', linestyle=':', alpha=0.5)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(sc1, cax=cbar_ax, label='Z Height (m)')
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(output_name, dpi=300)
    plt.close()
    print(f"Visualization saved to {output_name}")

if __name__ == "__main__":
    visualize_world('results/world_geometry_large.npy', 'results/master_world_large_viz.png')
    visualize_world('results/world_geometry.npy', 'results/master_world_small_viz.png')
