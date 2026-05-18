
import numpy as np

# Global Parameters for the VSLR System
# High-precision defaults verified in May 2026 benchmarks

# 1. Perception
ORB_NFEATURES = 2000

# 2. Peer-to-Peer Alignment (Stage 2)
SIGMA_D = 0.02           # Spatial Compatibility threshold (meters)
SPECTRAL_INLIER_RATIO = 0.8 # Top X% of principal cluster
RANSAC_ITERATIONS = 1000
INLIER_THRESHOLD = 0.1   # RMSE threshold for RANSAC (meters)

# 3. Global Synchronization (Stage 3)
GLOBAL_SIGMA_D = 0.02
IRLS_ITERATIONS = 25
EMA_ALPHA = 0.01
EMA_BETA = 0.98

# 4. Networking
TIMEOUT_SCAN = 10
TIMEOUT_POINTS = 30
