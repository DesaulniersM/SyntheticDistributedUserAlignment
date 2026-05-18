from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import os
import base64
import cv2
import sys

# Add parent path to allow finding common
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from environment.Camera import Camera

app = FastAPI()

# Configuration
MESH_PATH = os.getenv("MESH_PATH", "replica_office1.ply")
sensor = None

class ScanRequest(BaseModel):
    position: List[float]
    look_at: List[float]
    up: List[float]
    scan_id: Optional[str] = None 

class ScanResponse(BaseModel):
    points: List[List[float]]
    image_base64: Optional[str] = None
    depth_base64: Optional[str] = None # New: send depth map for feature mapping
    scan_id: Optional[str] = None

@app.on_event("startup")
def startup_event():
    global sensor
    path = MESH_PATH
    if not os.path.exists(path):
        # Local relative path fallback
        path = os.path.join(os.path.dirname(__file__), "..", MESH_PATH)
    
    print(f"Loading mesh from {path}")
    sensor = Camera(path)

@app.get("/health")
def health():
    return {"status": "ok", "mesh": MESH_PATH}

@app.post("/scan", response_model=ScanResponse)
def scan(request: ScanRequest):
    try:
        rgb_img, depth_map, points = sensor.capture(
            request.position, 
            request.look_at, 
            request.up
        )
        
        # 1. Encode RGB Image
        _, buffer_rgb = cv2.imencode('.png', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        rgb_str = base64.b64encode(buffer_rgb).decode('utf-8')
        
        # 2. Encode Depth Map (using 16-bit PNG for precision)
        # Scale to millimeters to fit in 16-bit uint
        depth_mm = (depth_map * 1000).astype(np.uint16)
        _, buffer_depth = cv2.imencode('.png', depth_mm)
        depth_str = base64.b64encode(buffer_depth).decode('utf-8')
        
        return ScanResponse(
            points=points.tolist(), 
            image_base64=rgb_str,
            depth_base64=depth_str,
            scan_id=request.scan_id
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
