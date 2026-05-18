# SMVR Algorithm Technical Summary (Li Fang et al., 2025)

This document summarizes the **Spectral Multiview Registration (SMVR)** pipeline as described in the paper: *"Robust Multiview Point Cloud Registration Using Algebraic Connectivity and Spatial Compatibility"* (IEEE TGRS, Vol. 63, 2025).

## Core Pipeline

### Stage 1: Sparse Graph Construction (Algebraic Connectivity)
The goal is to select a subset of pairs (edges) to align that ensures the global graph is "rigid" but avoids unnecessary (and potentially outlier-heavy) computations.

1.  **Feature Matching**: For each scan pair $(S_i, S_j)$, extract local descriptors (YOHO) and find mutual nearest neighbors to create a correspondence set $C_{ij}$.
2.  **Overlap Score Estimation**:
    *   Construct an adjacency matrix $A$ based on feature similarity: $A_{mn} = f_m \cdot f_n^T$.
    *   Compute the Laplacian: $Q = D - A$.
    *   **Algebraic Connectivity**: The overlap score is defined by the sum of algebraic connectivity (the second smallest eigenvalue $\lambda_2$ of $Q$) of the feature matching graphs.
3.  **Pruning**: Select the **Top-K** neighbors for each node based on the overlap score to form the sparse pose graph $\mathcal{G}$.

### Stage 2: Relative Registration Quality Analysis (Spatial Compatibility)
For the selected edges, compute a confidence weight $w_{ij}$ using spectral decomposition of a spatial compatibility matrix.

1.  **Spatial Compatibility Matrix ($M$):**
    For each correspondence pair $(a, b)$ in edge $(i, j)$, calculate the score based on distance preservation:
    $$SC(a, b) = \left[ 1 - \frac{(||p_{a} - p_{b}|| - ||q_{a} - q_{b}||)^2}{\theta_d^2} \right]_+$$
    *   $\theta_d$: Dividing value (Paper uses **0.01** for ScanNet/Indoor).
2.  **Spectral Decomposition:**
    Find the principal eigenvector $x$ of $M$ using power iteration ($x^{(k+1)} = Mx^{(k)}$).
3.  **Intercluster Score ($s$):**
    The confidence of the edge is $s = x^T M x$. This represents the reliability of the relative transformation $T_{ij}$ estimated from these correspondences.

### Stage 3: IRLS-based Global Pose Synchronization (HWA)
Refine the global poses $\{R_i, t_i\}$ iteratively by re-weighting edges to mitigate the impact of remaining outliers.

1.  **Synchronization Solve:**
    *   **Rotation:** Solve for global rotations $\{R_i\}$ by minimizing $\sum w_{ij} ||R_j - R_i R_{ij}||^2$ (Spectral Sync).
    *   **Translation:** Solve for global translations $\{t_i\}$ by minimizing $\sum w_{ij} ||t_j - (R_i t_{ij} + t_i)||^2$ (Least Squares).
2.  **Residual Calculation:**
    Compute the rotational residual $\delta_{ij}$ for each edge:
    $$\delta_{ij}^{(n)} = \text{angle}(\text{trace}(R_{ij}^T (R_i^T R_j)))$$
3.  **HWA (Historical Weighted Average) using EMA:**
    Instead of just using the current error, calculate an Exponential Moving Average of the error:
    $$v_{ij}^{(n+1)} = \lambda_\alpha (\delta_{ij}^{(n)} - \text{last\_err}) + \lambda_\beta v_{ij}^{(n)} + \lambda_\alpha \delta_{ij}^{(n)}$$
    *   **$\lambda_\alpha = 0.01$**, **$\lambda_\beta = 0.98$**.
4.  **Weight Update:**
    Update the edge weight for the next iteration:
    $$w_{ij}^{(n+1)} = \exp(-v_{ij}^{(n+1)})$$
    The final weight is the average of the weight history across all iterations.

## Standard Implementation Constants
*   **$\theta_d$ (Spatial Compat):** 0.01 (meters).
*   **EMA Alpha ($\lambda_\alpha$):** 0.01.
*   **EMA Beta ($\lambda_\beta$):** 0.98.
*   **Top-K Neighbors:** 3 to 10 (depending on density).
*   **IRLS Iterations:** 10.

## Mathematical Conventions (Refactored)
*   **Coordinate System:** Z-Up, Right-Handed (Robotics REP-103).
*   **Local Basis:** X-Forward, Y-Left, Z-Up.
*   **Composition Rule:** $T_{world\_j} = T_{world\_i} \cdot T_{ij}$.
