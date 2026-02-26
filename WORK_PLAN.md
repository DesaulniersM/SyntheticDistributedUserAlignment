# Work Plan: Porting Alignment Logic to Python

This plan outlines the steps to port the ICP and RANSAC alignment logic from the C# Unity project to the Python-based distributed system.

## 1. Environment Preparation
- [x] Add `scipy` to `requirements.txt`.
- [x] Rebuild Docker containers to include the new dependency. (Ready for user to run `docker-compose up --build`)

## 2. Porting Core Math (`SimpleICP.py`)
- [x] Implement `FindCorrespondences` using `scipy.spatial.cKDTree` for efficiency.
- [x] Implement `FindOptimalTransform` (SVD) using `numpy.linalg.svd`.
- [x] Port `Solve_RANSAC_4DoF_stick` for global registration.
- [x] Port `Solve_4DoF` and `Solve_4DoF_Anneal` for iterative refinement.
- [x] Port `Solve_Simple` (Standard ICP).
- [x] Implement `Solve_Robust` (RANSAC + ICP).

## 3. Porting Solver Logic (`AlignmentSolver.py`)
- [x] Port `RunConfiguredSolver` from `packetReciever.cs`.
- [x] Implement the decision hierarchy:
    - Gravity Fusion (if data available).
    - RANSAC Stick Matching (for global localization).
    - ICP/Annealing (for refinement).
- [x] Maintain solver state (last known transform) for temporal stability.

## 4. Integration into `MobileNode.py`
- [x] Add state variables to `MobileNode` to store the reference point cloud and last known transform.
- [x] Create a `/set_reference` endpoint.
- [x] Create a `/align` endpoint.

## 5. Verification & Testing
- [x] Create `test_alignment_local.py` to verify the math against known transforms.
- [x] Update `VisualizerClient.py` to demonstrate aligned clouds in Open3D.
- [ ] Run a multi-container test using `docker-compose`.
