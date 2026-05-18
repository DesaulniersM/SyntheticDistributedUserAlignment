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

### 5. Potential Venues
- **ICRA / IROS:** Leading robotics conferences (Focus on the decentralized/multi-user aspect).
- **ISMAR:** Top Augmented Reality conference (Focus on the low-power/mobile speedup).
- **IEEE TGRS:** High-impact journal (Focus on the spectral math and wavelet theory).
