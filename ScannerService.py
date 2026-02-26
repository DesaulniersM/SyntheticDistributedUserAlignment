from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from VirtualScanner import VirtualScanner
import os

app = FastAPI()

# Configuration
MESH_PATH = os.getenv("MESH_PATH", "labModel.obj")
scanner = None

class ScanRequest(BaseModel):
    position: List[float]
    look_at: List[float]
    up: List[float]

class ScanResponse(BaseModel):
    points: List[List[float]]

@app.on_event("startup")
def startup_event():
    global scanner
    if not os.path.exists(MESH_PATH):
        print(f"Warning: Mesh file {MESH_PATH} not found.")
    scanner = VirtualScanner(MESH_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "mesh": MESH_PATH}

@app.post("/scan", response_model=ScanResponse)
def scan(request: ScanRequest):
    try:
        points = scanner.scan_pose(
            request.position, 
            request.look_at, 
            request.up
        )
        # Convert numpy array to list for JSON serialization
        return ScanResponse(points=points.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
