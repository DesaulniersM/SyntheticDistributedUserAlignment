import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_master_world():
    # 1. Load data
    data_path = 'results/world_geometry.npy'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    data = np.load(data_path, allow_pickle=True).item()
    user_poses = data['user_poses']
    landmarks = data['landmarks']
    
    # Extract user positions (translation part of the 4x4 matrix)
    user_positions = user_poses[:, :3, 3]
    
    # 2. Create Figure
    fig = plt.figure(figsize=(18, 6), dpi=300)
    fig.suptitle('Master World Geometry Visualization', fontsize=16)
    
    # Panel 1: 3D Perspective View
    ax1 = fig.add_subplot(131, projection='3d')
    sc1 = ax1.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], 
                      c=landmarks[:, 2], cmap='viridis', s=1, alpha=0.5, label='Landmarks')
    ax1.scatter(user_positions[:, 0], user_positions[:, 1], user_positions[:, 2], 
                c='red', marker='o', s=10, label='User Positions')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Perspective View')
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 20)
    ax1.set_zlim(0, 2.5)
    ax1.legend(loc='upper right', markerscale=5)
    
    # Panel 2: Top-Down (XY) View
    ax2 = fig.add_subplot(132)
    sc2 = ax2.scatter(landmarks[:, 0], landmarks[:, 1], 
                      c=landmarks[:, 2], cmap='viridis', s=1, alpha=0.5)
    ax2.scatter(user_positions[:, 0], user_positions[:, 1], 
                c='red', marker='o', s=10, label='User Positions')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down (XY) View')
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 20)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', markerscale=5)
    
    # Panel 3: Side (XZ) View
    ax3 = fig.add_subplot(133)
    sc3 = ax3.scatter(landmarks[:, 0], landmarks[:, 2], 
                      c=landmarks[:, 2], cmap='viridis', s=1, alpha=0.5)
    ax3.scatter(user_positions[:, 0], user_positions[:, 2], 
                c='red', marker='o', s=10)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side (XZ) View - Vertical Stratification')
    ax3.set_xlim(0, 20)
    ax3.set_ylim(0, 2.5)
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # Add a horizontal lines for bands
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(y=1.5, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(y=2.0, color='gray', linestyle=':', alpha=0.5)
    
    # Annotate bands
    ax3.text(20.2, 0.25, 'Floor', verticalalignment='center')
    ax3.text(20.2, 1.15, 'Interaction', verticalalignment='center')
    ax3.text(20.2, 2.25, 'Ceiling', verticalalignment='center')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(sc1, cax=cbar_ax, label='Z Height (m)')
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    
    # 3. Save result
    save_path = 'results/master_world_viz.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    visualize_master_world()
