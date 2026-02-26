import requests
import open3d as o3d
import numpy as np
import time

def rotate_vector(vector, angle_degrees, axis='y'):
    """Helper to rotate a vector around an axis."""
    rad = np.radians(angle_degrees)
    if axis == 'y':
        matrix = np.array([
            [ np.cos(rad), 0, np.sin(rad)],
            [ 0,           1, 0          ],
            [-np.sin(rad), 0, np.cos(rad)]
        ])
    return matrix @ np.array(vector)

def get_view_matrix(position, look_at, up):
    """Calculates the world-to-local matrix (View Matrix)."""
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

def create_camera_frustum(view_matrix, color, scale=1.0):
    pts = np.array([
        [0, 0, 0],
        [-0.5, -0.5, 1], [0.5, -0.5, 1],
        [0.5, 0.5, 1], [-0.5, 0.5, 1]
    ]) * scale
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    ls.transform(np.linalg.inv(view_matrix))
    return ls

def run_visualization_pipeline():
    print("--- Starting Comparative Visualization (6x Scale Aware) ---")
    look_at = [1.24, 2.85, -1.13]
    up = [0, 1, 0] # Y-UP
    pos_2 = [8.24, 11.61, -9.13]
    
    # Node 1 offset
    vec = np.array(pos_2) - np.array(look_at)
    rotated_vec = rotate_vector(vec, 30, axis='y')
    pos_1 = (np.array(look_at) + rotated_vec).tolist()
    
    v2 = get_view_matrix(pos_2, look_at, up)
    v1 = get_view_matrix(pos_1, look_at, up)
    gt_t_1_to_2 = v2 @ np.linalg.inv(v1)

    print("Requesting local scans...")
    try:
        t_pts_local = np.array(requests.post("http://localhost:8000/scan", json={"position": pos_2, "look_at": look_at, "up": up}).json()["points"])
        s_pts_local = np.array(requests.post("http://localhost:8000/scan", json={"position": pos_1, "look_at": look_at, "up": up}).json()["points"])
    except Exception as e:
        print(f"Scans failed: {e}")
        return

    print("Aligning...")
    requests.post("http://localhost:8002/set_reference", json={"points": t_pts_local.tolist()})
    local_gravity = [0, -1, 0] 
    resp = requests.post("http://localhost:8002/align", json={
        "source_points": s_pts_local.tolist(), 
        "host_gravity": local_gravity, 
        "local_gravity": local_gravity
    })
    
    found_t_1_to_2 = np.array(resp.json()["transform"])

    print(f"\nDiagnostics:")
    gt_yaw = np.degrees(np.arctan2(gt_t_1_to_2[0, 2], gt_t_1_to_2[0, 0]))
    found_yaw = np.degrees(np.arctan2(found_t_1_to_2[0, 2], found_t_1_to_2[0, 0]))
    print(f"GT Yaw: {gt_yaw:.2f} deg | Found Yaw: {found_yaw:.2f} deg")

    # Geometry Setup - Offset increased for 17m room
    r1_off = np.array([-25, 0, 0])
    r2_off = np.array([25, 0, 0])
    
    def prep_room(offset):
        m = o3d.io.read_triangle_mesh("labModel.obj")
        m.triangle_material_ids = o3d.utility.IntVector()
        m.paint_uniform_color([0.8, 0.8, 0.8])
        m.translate(offset)
        return m

    # Room 1
    p_ref_1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(t_pts_local))
    p_ref_1.transform(np.linalg.inv(v2))
    p_ref_1.translate(r1_off)
    p_ref_1.paint_uniform_color([0, 0.8, 0])

    p_src_1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s_pts_local))
    p_src_1.transform(np.linalg.inv(v1))
    p_src_1.translate(r1_off)
    p_src_1.paint_uniform_color([0.8, 0, 0])

    # Room 2
    p_ref_2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(t_pts_local))
    p_ref_2.transform(np.linalg.inv(v2))
    p_ref_2.translate(r2_off)
    p_ref_2.paint_uniform_color([0, 0.8, 0])

    p_aligned_2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s_pts_local))
    p_aligned_2.transform(found_t_1_to_2)
    p_aligned_2.transform(np.linalg.inv(v2))
    p_aligned_2.translate(r2_off)
    p_aligned_2.paint_uniform_color([0, 0.5, 1.0])

    # Frustums - Scaled up for room size
    cam_scale = 3.0
    f2_1 = create_camera_frustum(v2, [0, 1, 0], cam_scale); f2_1.translate(r1_off)
    f1_1 = create_camera_frustum(v1, [1, 0, 0], cam_scale); f1_1.translate(r1_off)
    f2_2 = create_camera_frustum(v2, [0, 1, 0], cam_scale); f2_2.translate(r2_off)
    f1_res = create_camera_frustum(np.linalg.inv(found_t_1_to_2) @ v2, [0, 0.5, 1], cam_scale); f1_res.translate(r2_off)

    o3d.visualization.draw([
        {'name': 'R1: Lab', 'geometry': prep_room(r1_off)},
        {'name': 'R1: Ref', 'geometry': p_ref_1}, {'name': 'R1: Src', 'geometry': p_src_1},
        {'name': 'R1: Ref Cam', 'geometry': f2_1}, {'name': 'R1: Src Cam', 'geometry': f1_1},
        {'name': 'R2: Lab', 'geometry': prep_room(r2_off)},
        {'name': 'R2: Ref', 'geometry': p_ref_2}, {'name': 'R2: Aligned', 'geometry': p_aligned_2},
        {'name': 'R2: Ref Cam', 'geometry': f2_2}, {'name': 'R2: Aligned Cam', 'geometry': f1_res}
    ], show_ui=True)

if __name__ == "__main__":
    run_visualization_pipeline()
