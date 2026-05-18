from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import uvicorn
import base64
import cv2
import numpy as np
from typing import List, Optional, Dict
import sys

# Add parent path to allow finding solvers and common
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from common.visual_features import VisualFeatureEngine
    from solvers.local_solvers.VisualProcrustesSolver import VisualProcrustesSolver
    from solvers.local_solvers.AlignmentSolver import AlignmentSolver
except ImportError:
    sys.path.append('/app')
    from common.visual_features import VisualFeatureEngine
    from solvers.local_solvers.VisualProcrustesSolver import VisualProcrustesSolver
    from solvers.local_solvers.AlignmentSolver import AlignmentSolver

app = FastAPI()

class MobileNode:
    def __init__(self, node_id, scanner_url):
        self.node_id = node_id
        self.scanner_url = scanner_url
        self.feature_engine = VisualFeatureEngine()
        self.visual_solver = VisualProcrustesSolver()
        
        self.local_points = None
        self.local_descriptors = None
        self.local_3d_features = None
        self.local_wavelet_gists = None # New experimental memory

node = None

@app.on_event("startup")
def startup():
    global node
    node_id = os.getenv("NODE_ID", "node1")
    scanner_url = os.getenv("SCANNER_URL", "http://scanner:8000")
    node = MobileNode(node_id, scanner_url)

@app.post("/scan")
def do_scan(pos: List[float], look: List[float], up: List[float], use_wavelets: bool = False):
    try:
        resp = requests.post(f"{node.scanner_url}/scan", json={
            "position": pos, "look_at": look, "up": up, "scan_id": node.node_id
        })
        data = resp.json()
        node.local_points = np.array(data["points"])
        
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data["image_base64"]), np.uint8), cv2.IMREAD_COLOR)
        depth_img = cv2.imdecode(np.frombuffer(base64.b64decode(data["depth_base64"]), np.uint8), cv2.IMREAD_UNCHANGED)
        depth_map = depth_img.astype(np.float32) / 1000.0
        
        # Enable wavelet mode if requested
        node.feature_engine.use_wavelets = use_wavelets
        des, pts3d, gists = node.feature_engine.extract_3d_features(img, depth_map)
        
        node.local_descriptors = des
        node.local_3d_features = pts3d
        node.local_wavelet_gists = gists
        
        return {
            "status": "success", 
            "landmarks": len(pts3d) if pts3d is not None else 0,
            "wavelets": use_wavelets,
            "image_base64": data["image_base64"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features")
def get_features():
    if node.local_descriptors is None:
        raise HTTPException(status_code=404, detail="No features available.")
    
    return {
        "descriptors": node.local_descriptors.tolist(),
        "points_3d": node.local_3d_features.tolist(),
        "wavelet_gists": node.local_wavelet_gists.tolist() if node.local_wavelet_gists is not None else None
    }

@app.get("/points")
def get_points():
    if node.local_points is None: return {"points": []}
    return {"points": node.local_points.tolist()}

@app.post("/align-with-peer")
def align_with_peer(peer_url: str, use_wavelets: bool = False):
    try:
        resp = requests.get(f"{peer_url}/features")
        peer_data = resp.json()
        
        peer_des = np.array(peer_data["descriptors"], dtype=np.uint8)
        peer_pts = np.array(peer_data["points_3d"])
        peer_gists = np.array(peer_data["wavelet_gists"]) if peer_data.get("wavelet_gists") else None
        
        # Use gists if both sides have them and we are in wavelet mode
        active_gists = node.local_wavelet_gists if use_wavelets else None
        active_peer_gists = peer_gists if use_wavelets else None

        T, confidence = node.visual_solver.solve(
            node.local_descriptors, node.local_3d_features,
            peer_des, peer_pts,
            gists_src=active_gists, gists_tgt=active_peer_gists
        )
        
        return {"transform": T.tolist(), "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
