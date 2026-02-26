from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import uvicorn
import socket
import threading
import json
import numpy as np
from typing import List, Optional

# --- New Imports for Alignment ---
from AlignmentSolver import AlignmentSolver

app = FastAPI()

# Configuration from environment variables
SCANNER_URL = os.getenv("SCANNER_URL", "http://scanner:8000")
NODE_ID = os.getenv("NODE_ID", "unknown")
UDP_PORT = int(os.getenv("UDP_PORT", 9000))

# --- Global State for Alignment ---
solver = AlignmentSolver()
reference_points: Optional[np.ndarray] = None

class Pose(BaseModel):
    position: List[float]
    look_at: List[float]
    up: List[float]

class Message(BaseModel):
    sender: str
    content: str
    data: Optional[dict] = None

class AlignRequest(BaseModel):
    source_points: List[List[float]]
    host_gravity: Optional[List[float]] = None
    local_gravity: Optional[List[float]] = None

# --- UDP Logic ---
def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    print(f"UDP Listener started on port {UDP_PORT}")
    while True:
        data, addr = sock.recvfrom(4096)
        try:
            msg = json.loads(data.decode())
            print(f"UDP received from {addr}: {msg}", flush=True)
        except Exception as e:
            print(f"UDP Error: {e}", flush=True)

@app.on_event("startup")
def start_udp():
    thread = threading.Thread(target=udp_listener, daemon=True)
    thread.start()

class UDPSendRequest(BaseModel):
    target_host: str
    target_port: int
    content: str
    data: Optional[dict] = None

@app.post("/send_udp")
def trigger_udp(req: UDPSendRequest):
    try:
        send_udp(req.target_host, req.target_port, req.content, req.data)
        return {"status": "UDP sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def send_udp(target_host: str, target_port: int, content: str, data: dict = None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    msg = {"sender": NODE_ID, "content": content, "data": data}
    sock.sendto(json.dumps(msg).encode(), (target_host, target_port))

# --- HTTP Logic ---
@app.get("/")
def read_root():
    return {"node_id": NODE_ID, "status": "active"}

@app.post("/request_scan")
def request_scan(pose: Pose):
    """Request a synthetic point cloud from the ScannerService."""
    try:
        response = requests.post(f"{SCANNER_URL}/scan", json=pose.dict())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scan: {str(e)}")

@app.post("/receive_message")
def receive_message(msg: Message):
    """Interface for other nodes to send data to this node."""
    print(f"Received message from {msg.sender}: {msg.content}", flush=True)
    return {"status": "message_received"}

# --- Alignment Endpoints ---

@app.post("/set_reference")
def set_reference(req: dict):
    """Sets the local reference point cloud for future alignments."""
    global reference_points
    try:
        reference_points = np.array(req["points"])
        return {"status": "Reference set", "point_count": len(reference_points)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/align")
def align_pc(req: AlignRequest):
    """Aligns incoming points with the stored reference cloud."""
    global reference_points
    if reference_points is None:
        raise HTTPException(status_code=400, detail="Reference points not set. Call /set_reference first.")
    
    try:
        host_gravity = np.array(req.host_gravity) if req.host_gravity else None
        local_gravity = np.array(req.local_gravity) if req.local_gravity else None
        
        transform, error = solver.run_configured_solver(
            req.source_points, 
            reference_points, 
            host_gravity, 
            local_gravity
        )
        
        return {
            "transform": transform.tolist(),
            "error": float(error),
            "status": "success"
        }
    except Exception as e:
        print(f"Alignment error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transform_pc")
def transform_pc(req: dict):
    """Receive points and return them. (Bias removed for realistic local-space testing)"""
    try:
        points = np.array(req["points"])
        if points.size == 0:
            return {"points": [], "node_id": NODE_ID}
        
        print(f"Node {NODE_ID} passing through {len(points)} points.", flush=True)
        return {"points": points.tolist(), "node_id": NODE_ID}
    except Exception as e:
        print(f"Transform error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
