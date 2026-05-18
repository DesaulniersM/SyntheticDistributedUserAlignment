import cv2
import numpy as np
import os
import sys

# Optional experimental import
try:
    from common.wavelet_utils import HaarWaveletExperiment
except ImportError:
    # Handle direct execution or docker
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from common.wavelet_utils import HaarWaveletExperiment

try:
    from config.settings import ORB_NFEATURES
except ImportError:
    # Handle docker path
    sys.path.append('/app')
    from config.settings import ORB_NFEATURES

class VisualFeatureEngine:
    """
    Handles 2D ORB feature extraction and 3D back-projection.
    """
    def __init__(self, fov=60.0, width=640, height=480, use_wavelets=False):
        self.orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
        self.use_wavelets = use_wavelets
        
        # Standard Intrinsics
        f = (width / 2) / np.tan(np.radians(fov / 2))
        self.K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
        self.width = width; self.height = height

    def extract_3d_features(self, rgb_img, depth_map):
        if len(rgb_img.shape) == 3:
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb_img
            
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None: return None, None
            
        points_3d = []
        wavelet_gists = []
        valid_indices = []
        
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        for i, k in enumerate(kp):
            u, v = int(k.pt[0]), int(k.pt[1])
            z = depth_map[v, u]
            
            if z > 0:
                # 1. 3D Back-projection
                lx, ly, lz = (u - cx) * z / fx, (v - cy) * z / fy, z
                points_3d.append([lz, -lx, -ly]) # Robotics frame
                
                # 2. Experimental Wavelet Gist (if enabled)
                if self.use_wavelets:
                    # Extract 16x16 patch around keypoint
                    x1, y1 = max(0, u-8), max(0, v-8)
                    x2, y2 = min(self.width, u+8), min(self.height, v+8)
                    patch = gray[y1:y2, x1:x2]
                    # Resize to exactly 16x16 if at edges
                    if patch.shape != (16, 16):
                        patch = cv2.resize(patch, (16, 16))
                    
                    gist = HaarWaveletExperiment.get_gist(patch, levels=2)
                    wavelet_gists.append(gist)
                
                valid_indices.append(i)
                
        if not points_3d: return None, None
            
        final_des = des[valid_indices]
        if self.use_wavelets:
            return final_des, np.array(points_3d), np.array(wavelet_gists)
            
        return final_des, np.array(points_3d), None

    @staticmethod
    def match_features(des1, des2, use_wavelets=False):
        if use_wavelets:
            # High-speed Euclidean matching on sparse wavelet gists
            # des1, des2 are now [N, 16] float arrays
            # We'll use a simple distance matrix for this experiment
            dists = np.linalg.norm(des1[:, np.newaxis] - des2, axis=2)
            idx_src = np.arange(len(des1))
            idx_tgt = np.argmin(dists, axis=1)
            
            # Simple mutual check for robustness
            idx_tgt_back = np.argmin(dists, axis=0)
            
            matches = []
            for i, j in enumerate(idx_tgt):
                if idx_tgt_back[j] == i and dists[i, j] < 50.0: # Threshold for gist
                    # Mock a cv2.DMatch object for compatibility
                    class Match:
                        def __init__(self, q, t, d):
                            self.queryIdx = q; self.trainIdx = t; self.distance = d
                    matches.append(Match(i, j, dists[i, j]))
            return matches

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)
