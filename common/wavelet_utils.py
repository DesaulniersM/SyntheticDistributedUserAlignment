import numpy as np

class HaarWaveletExperiment:
    """
    Experimental utility for 2D Haar Wavelet Transforms.
    Used for landmark descriptor compression/denoising.
    """
    
    @staticmethod
    def transform_2d(patch):
        """
        Performs a 1-level 2D Haar DWT on a square patch (e.g., 16x16).
        Returns: (Approximation, Horizontal, Vertical, Diagonal)
        """
        # Ensure patch is 2D and even-sized
        h, w = patch.shape
        # Average and Difference (Haar basis)
        # s1 = (a+b)/2, d1 = (a-b)/2
        
        # Horizontal pass
        h_row = (patch[:, 0::2] + patch[:, 1::2]) / 2.0
        h_diff = (patch[:, 0::2] - patch[:, 1::2]) / 2.0
        
        # Vertical pass on the horizontal results
        appx = (h_row[0::2, :] + h_row[1::2, :]) / 2.0
        vert = (h_row[0::2, :] - h_row[1::2, :]) / 2.0
        horz = (h_diff[0::2, :] + h_diff[1::2, :]) / 2.0
        diag = (h_diff[0::2, :] - h_diff[1::2, :]) / 2.0
        
        return appx, horz, vert, diag

    @staticmethod
    def get_gist(patch, levels=2):
        """
        Reduces a 16x16 patch to a very sparse 'Gist' vector (the Approximation coefficients).
        """
        current = patch.astype(np.float32)
        for _ in range(levels):
            # We only care about the LL (Approximation) for a sparse gist
            current, _, _, _ = HaarWaveletExperiment.transform_2d(current)
        
        # Return flattened coefficients as a sparse descriptor
        return current.flatten()
