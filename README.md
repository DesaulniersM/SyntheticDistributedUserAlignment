# Point Cloud Alignment Simulator

This project simulates a distributed system of mobile devices (Nodes) that perform peer-to-peer 4-DoF point cloud alignment using synthetic data generated from a 3D mesh.

## System Architecture

- **Scanner Service (`ScannerService.py`)**: A central service that simulates a depth camera by raycasting against `labModel.obj`. It uses a **Geometric Edge Sampling** strategy to mimic realistic mobile AR feature points.
- **Mobile Nodes (`MobileNode.py`)**: Distributed agents that request scans, store reference frames, and perform alignments.
- **Alignment Engines**:
    - **SimpleICP**: Core 4-DoF (Yaw + 3D Translation) solver using RANSAC and SVD-based ICP.
    - **Incremental (BFS)**: Path-based propagation of transforms from an anchor node. Stable for small groups.
    - **Spectral (1D)**: Angular Synchronization using complex eigenvectors to solve for global consensus yaws. Robust against drift in larger groups.

## Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.10+ (for local testing)

### 1. Start the Environment
The simulation environment must be running in Docker for the network nodes to communicate.

```bash
docker-compose up -d --build
```
This starts:
- `scanner` at `http://localhost:8000`
- `node1` at `http://localhost:8001`
- `node2` at `http://localhost:8002`

## Running Tests

### 1. Local Math Verification
Verify the core SimpleICP logic without network overhead using synthetic L-shaped structures.
```bash
python3 test_alignment_local.py
```

### 2. Headless Service Test
Verify that the Docker containers are communicating correctly and can perform a basic two-node alignment.
```bash
python3 test_alignment_headless.py
```

### 3. Multi-User Incremental Alignment (BFS)
Tests the propagation of poses through a sparse graph of 5 users using the BFS approach.
```bash
python3 TestGlobalAlignment.py
```

### 4. Multi-User Spectral Synchronization
Tests the global consensus solver using 1D Angular Synchronization for 6 users.
```bash
python3 TestSpectralAlignment.py
```

## Key Files
- `SimpleICP.py`: The core mathematical alignment logic.
- `VirtualScanner.py`: Handles raycasting and edge-prioritized sampling.
- `MultiUserAlignment.py`: Contains the `SpectralAlignmentManager`.
- `IncrementalAlignment.py`: Contains the `GlobalAlignmentManager` (BFS).
- `VisualizerClient.py`: A comprehensive pipeline that runs a scan, aligns, and opens an Open3D window.

## Developer Notes
- **Scaling**: The `SimpleICP.SCALE` constant is set to `6.0` to account for the large dimensions of the `labModel.obj` mesh.
- **Edge Sampling**: The scanner is currently configured to sample 50% from geometric edges and 50% from flat surfaces to balance feature detection and structural alignment.
