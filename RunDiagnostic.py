import open3d as o3d
import numpy as np
from VirtualScanner import VirtualScanner  # Assuming your previous file is named scanner.py

def run_diagnostic(obj_path):
    print(f"--- DIAGNOSTIC FOR: {obj_path} ---")
    
    # 1. Load and Inspect Mesh Geometry
    try:
        mesh = o3d.io.read_triangle_mesh(obj_path)
        # Check if we actually loaded triangles
        tri_count = len(mesh.triangles)
        vert_count = len(mesh.vertices)
        
        print(f"Vertices:  {vert_count}")
        print(f"Triangles: {tri_count}")
        
        if tri_count == 0:
            print("\n[CRITICAL FAILURE] The mesh has 0 triangles!")
            print("Fix: In Blender Export settings, check 'Triangulate Faces'.")
            print("Fix: Ensure you are exporting 'Mesh', not 'Armature' or 'Empty'.")
            return

        # 2. Check Scale and Position
        min_bound = mesh.get_min_bound()
        max_bound = mesh.get_max_bound()
        center = mesh.get_center()
        size = max_bound - min_bound
        
        print(f"\nBounds Min: {min_bound}")
        print(f"Bounds Max: {max_bound}")
        print(f"Center:     {center}")
        print(f"Size:       {size}")
        
        # 3. Perform "Auto-Aimed" Scan
        # We place the camera at the Center + slight offset in Y and Z to ensure we are inside
        print("\n--- ATTEMPTING AUTO-AIMED SCAN ---")
        
        scanner = VirtualScanner(obj_path)
        
        # Position camera at the center of the room bounds
        # But back up slightly so we aren't inside a distinct object
        cam_pos = center + [0, size[1] * 0.2, 0] 
        
        # Look slightly down/forward
        look_at = center + [0, 0, 1] 
        up = [0, 1, 0]
        
        print(f"Cam Pos: {cam_pos}")
        print(f"Looking At: {look_at}")
        
        points = scanner.scan_pose(cam_pos, look_at, up)
        
        print(f"\n[RESULT] Generated cloud with {len(points)} points.")
        
        if len(points) > 0:
            print("SUCCESS! The logic works, but your previous coordinates were wrong.")
        else:
            print("STILL 0 POINTS. Possible Backface Culling issue or Camera is inside a solid object.")

    except Exception as e:
        print(f"Diagnostic crashed: {e}")

if __name__ == "__main__":
    run_diagnostic("labModel.obj")