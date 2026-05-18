import numpy as np

def get_view_matrix(position, look_at, up=np.array([0, 0, 1])):
    """Calculates the world-to-local matrix (View Matrix) for Open3D."""
    z_axis = np.array(look_at) - np.array(position)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(np.array(up), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    r = np.vstack([x_axis, y_axis, z_axis])
    t = -r @ np.array(position)
    view_mat = np.eye(4)
    view_mat[:3, :3] = r
    view_mat[:3, 3] = t
    return view_mat

def rotate_vector_z(vector, deg):
    """Rotates a vector around the Z-axis."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R @ np.array(vector)

def get_robotic_pose(position, look_at, world_up=np.array([0, 0, 1])):
    """
    Creates a 4x4 matrix T such that P_world = T @ P_local.
    Local Frame: X-Forward, Y-Left, Z-Up.
    """
    pos = np.array(position)
    look = np.array(look_at)
    
    # 1. Forward direction (X_local)
    fwd = look - pos
    fwd = fwd / np.linalg.norm(fwd)
        
    # 2. Left direction (Y_local)
    # Right-handed: Z x X = Y => Up x Forward = Left
    left = np.cross(world_up, fwd)
    left = left / np.linalg.norm(left)
        
    # 3. Up direction (Z_local)
    up = np.cross(fwd, left)
    
    # R basis as columns
    R = np.vstack([fwd, left, up]).T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
    return T

def world_to_local(pts_world, T_world_local):
    """P_local = T^-1 @ P_world"""
    pts_world = np.array(pts_world)
    R = T_world_local[:3, :3]
    t = T_world_local[:3, 3]
    # P_loc = R^T @ (P_world - t)
    return (R.T @ (pts_world - t).T).T

def local_to_world(pts_local, T_world_local):
    """P_world = T @ P_local"""
    pts_local = np.array(pts_local)
    R = T_world_local[:3, :3]
    t = T_world_local[:3, 3]
    return (R @ pts_local.T).T + t

def get_yaw_from_matrix(T):
    """Extracts Z-Yaw from a Robotics Pose matrix."""
    R = T[:3, :3]
    return np.arctan2(R[1, 0], R[0, 0])
