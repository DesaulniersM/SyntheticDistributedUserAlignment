# Research Plan: VSLR-Sync
## Toward Real-Time Distributed Visual-Spectral Registration for Mobile AR

This document outlines the roadmap to transition the current prototype into a publishable academic manuscript.

### 1. Research Objectives
- **O1: Computational Efficiency.** Quantify the speedup provided by Wavelet-Gist landmark compression in high-density multi-user environments.
- **O2: Bandwidth Optimization.** Prove that the hybrid visual-spectral pipeline minimizes network overhead for mobile agents.
- **O3: Robustness to Noise.** Validate the effectiveness of Stage 2 Spectral Compatibility and Stage 3 HWA in high-outlier (noisy) scenarios.

### 2. Experimental Design

#### Experiment A: Scalability & Quadratic Growth
- **Setup:** Launch Docker clusters of sizes N=[2, 6, 12, 24, 48].
- **Comparison:** Compare matching time of standard ORB descriptors vs. Haar-Wavelet Gists.
- **Hypothesis:** Wavelet Gists will show a significantly lower slope in processing time as the number of nodes increases.

#### Experiment B: Bandwidth & Data Footprint
- **Setup:** Measure the byte-size of the data packets required for alignment.
- **Baseline 1:** Full Point Cloud (Standard ICP).
- **Baseline 2:** Full ORB Descriptors (Standard Visual SLAM).
- **Proposed:** Sparse Wavelet Gists.
- **Metric:** `Total Bytes Transferred per Edge`.

#### Experiment C: Robustness Ablation Study
- **Setup:** Use `validate_smvr.py` to inject increasing outlier percentages (0%, 20%, 40%, 60%).
- **Comparison:** Compare Global RMSE of (1) Standard RANSAC, (2) Spectral Sync without HWA, and (3) Full VSLR-Sync.
- **Metric:** `Breakdown Point` (The noise level at which RMSE exceeds 0.5m).

### 3. Key Metrics for Publication
1. **Mean Dataset RMSE (m):** Geometric accuracy of the final map.
2. **Mean Time per User (s):** Latency from scan to global alignment.
3. **Data Efficiency Ratio:** Bandwidth saved compared to traditional methods.

### 4. Target Datasets
- **Replica Dataset:** (Current) For clean, high-fidelity office environments.
- **ScanNet:** For diverse, real-world indoor messy scenes.
- **Matterport3D:** For large-scale, multi-room building scans.

### 6. Standard of Rigor for ACM/IEEE Publication
To ensure submission-ready results for high-impact venues (ICRA, IROS, ISMAR), the following methodological standards are adopted:

1. **Standardized Baselines:** All iterative optimization results must be compared against industry-standard solvers (**GTSAM** or **Ceres**) rather than custom scripts. This prevents "strawman" comparisons and ensures scientific fairness.
2. **Realistic Sensing Constraints:**
   - **Heading-Limited FOV:** Agents must only observe landmarks within a 90° horizontal Field of View, creating realistic asymmetric overlaps.
   - **Clustered Geometry:** Landmarks are distributed as discrete objects (clustered and stratified) rather than uniform "fog" to test structural sensitivity.
3. **Advanced Robotics Metrics:**
   - **APE (Absolute Pose Error):** To measure global map consistency.
   - **RPE (Relative Pose Error):** To measure local drift between neighbors.
   - **Success Rate Statistics:** Percentage of trials achieving <5cm convergence across N trials.
4. **Efficiency Frontier:** Generation of Pareto charts (Latency vs. Accuracy) to demonstrate the algorithm's position as the optimal trade-off for battery-powered AR devices.

### 7. Decentralization Architecture & Scientific Novelty
As the system transitions from a simulated single-node testbed to a true distributed multi-robot deployment, the architectural advantages of the Spectral method over iterative solvers (D-PGO) become a central narrative for publication.

#### The Communication Bottleneck in D-PGO
Traditional Distributed Pose Graph Optimization (D-PGO) methods (e.g., decentralized GTSAM or ADMM-based iterative solvers) rely on the exchange of Separator Marginals or gradient updates. 
- **The Iterative Penalty:** These methods require robots to broadcast dense matrices to their neighbors at *every iteration* of the optimization.
- **Latency Vulnerability:** In a sparse network, the total time to convergence is dictated by network latency and the number of communication rounds, not CPU speed. A 50-iteration solve requires 50 sequential network hops.

#### The Spectral Advantage
The proposed Spectral Synchronization architecture is highly amenable to bandwidth-constrained environments:
- **Phase 1: Local Peer-to-Peer Filtering (Fully Decentralized):** The construction of the Spatial Compatibility Matrix ($M$) and its principal eigenvector occurs exclusively between pairs of neighboring robots. This step is independent and parallelizes perfectly across the swarm.
- **Phase 2: Global Relaxation (Single-Shot Sync):** Unlike iterative methods, Spectral Synchronization formulates the global alignment as a direct Eigen-problem on the Connection Laplacian ($H$). In a decentralized setting, this can be resolved via Distributed Power Iteration (Gossip algorithms) or passed to a central compute node. Because it is a non-iterative, global relaxation, it avoids the "Cold Start" local minima traps that plague D-PGO in sparse networks.

#### State-of-the-Art Novelty Context
Has this been done before?
- **SE-Sync (Rosen et al., 2017) & Certifiably Correct PGO:** SE-Sync uses Riemannian optimization and spectral relaxations to find the globally optimal pose graph. It is the gold standard for avoiding local minima in SLAM.
- **Kimera-Multi & Distributed PGO:** State-of-the-art multi-robot systems have begun implementing distributed versions of these robust solvers.
- **Your Novelty:** Your specific contribution is the **Pre-Synchronization Pruning via Algebraic Connectivity**. Methods like SE-Sync assume the edges of the pose graph are somewhat reliable. Your Stage 2 uses the Spatial Compatibility Matrix's eigenvector to assign *confidence weights* (or delete edges entirely) based on geometric rigidity *before* the global synchronization occurs. The combination of local spectral pruning (robustness to outliers) with global spectral synchronization (robustness to initialization) in extremely sparse, large-scale networks is a compelling and highly publishable architectural contribution.
