"""
Stain Normalization for Digital Pathology

Implements Macenko and Reinhard methods for H&E stain normalization
to handle scanner/staining variation across sites.
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StainNormalizer:
    """Base class for stain normalization methods."""
    
    def __init__(self):
        self.target_stains = None
        self.target_concentrations = None
    
    def fit(self, target_image: np.ndarray) -> "StainNormalizer":
        """Fit normalizer to target image."""
        raise NotImplementedError
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to target."""
        raise NotImplementedError
    
    def fit_transform(self, target: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(target)
        return self.transform(image)


class MacenkoNormalizer(StainNormalizer):
    """
    Macenko stain normalization for H&E images.
    
    Reference: Macenko et al. "A method for normalizing histology slides 
    for quantitative analysis." ISBI 2009.
    """
    
    def __init__(self, luminosity_threshold: float = 0.8, alpha: float = 1.0, beta: float = 0.15):
        super().__init__()
        self.luminosity_threshold = luminosity_threshold
        self.alpha = alpha  # Percentile for max concentrations
        self.beta = beta    # OD threshold
        
    def _rgb_to_od(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to optical density."""
        rgb = rgb.astype(np.float32) + 1  # Avoid log(0)
        od = -np.log(rgb / 255.0)
        return od
    
    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert optical density to RGB."""
        rgb = 255 * np.exp(-od)
        return np.clip(rgb, 0, 255).astype(np.uint8)
    
    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix rows to unit length."""
        return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    
    def _get_stain_matrix(self, od: np.ndarray) -> np.ndarray:
        """Extract stain matrix using SVD."""
        # Remove background (low OD pixels)
        od_flat = od.reshape(-1, 3)
        od_hat = od_flat[(od_flat > self.beta).any(axis=1)]
        
        if len(od_hat) == 0:
            logger.warning("No tissue pixels found, using default stain matrix")
            # Default H&E stain matrix
            return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
        
        # Compute eigenvectors via SVD
        _, _, V = np.linalg.svd(od_hat, full_matrices=False)
        
        # Project on plane spanned by first 2 eigenvectors
        V = V[:2]
        
        # Ensure correct angle (H&E specific)
        if V[0, 0] < 0:
            V[0] = -V[0]
        if V[1, 0] < 0:
            V[1] = -V[1]
            
        # Find extreme angles
        angles = np.arctan2(od_hat @ V[1], od_hat @ V[0])
        min_angle = np.percentile(angles, 100 * (1 - self.alpha))
        max_angle = np.percentile(angles, 100 * self.alpha)
        
        # Compute stain vectors
        v_min = np.array([np.cos(min_angle), np.sin(min_angle)])
        v_max = np.array([np.cos(max_angle), np.sin(max_angle)])
        
        # Convert back to OD space
        stain_matrix = np.array([v_min @ V, v_max @ V])
        
        return self._normalize_rows(stain_matrix)
    
    def _get_concentrations(self, od: np.ndarray, stains: np.ndarray) -> np.ndarray:
        """Get stain concentrations via least squares."""
        od_flat = od.reshape(-1, 3)
        concentrations = np.linalg.lstsq(stains.T, od_flat.T, rcond=None)[0].T
        return concentrations.reshape(od.shape[:2] + (2,))
    
    def fit(self, target_image: np.ndarray) -> "MacenkoNormalizer":
        """Fit to target image."""
        target_od = self._rgb_to_od(target_image)
        self.target_stains = self._get_stain_matrix(target_od)
        self.target_concentrations = self._get_concentrations(target_od, self.target_stains)
        
        # Store max concentrations for normalization
        self.max_c_target = np.percentile(
            self.target_concentrations.reshape(-1, 2), 
            99, 
            axis=0
        )
        return self
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to target."""
        if self.target_stains is None:
            raise ValueError("Must call fit() before transform()")
        
        # Get source stain matrix and concentrations
        source_od = self._rgb_to_od(image)
        source_stains = self._get_stain_matrix(source_od)
        source_concentrations = self._get_concentrations(source_od, source_stains)
        
        # Normalize concentrations
        max_c_source = np.percentile(
            source_concentrations.reshape(-1, 2), 
            99, 
            axis=0
        )
        source_concentrations *= (self.max_c_target / (max_c_source + 1e-10))
        
        # Reconstruct with target stains
        normalized_od = source_concentrations @ self.target_stains
        normalized_rgb = self._od_to_rgb(normalized_od)
        
        return normalized_rgb


class ReinhardNormalizer(StainNormalizer):
    """
    Reinhard color normalization.
    
    Reference: Reinhard et al. "Color transfer between images." 
    IEEE Computer Graphics and Applications 2001.
    """
    
    def __init__(self):
        super().__init__()
        self.target_means = None
        self.target_stds = None
    
    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to LAB color space."""
        # Normalize to [0, 1]
        rgb = rgb.astype(np.float32) / 255.0
        
        # RGB to XYZ
        mask = rgb > 0.04045
        rgb[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
        rgb[~mask] /= 12.92
        
        # XYZ transformation matrix
        M = np.array([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ])
        
        xyz = rgb @ M.T
        
        # XYZ to LAB
        xyz /= np.array([0.95047, 1.0, 1.08883])  # D65 illuminant
        
        mask = xyz > 0.008856
        xyz[mask] = np.power(xyz[mask], 1/3)
        xyz[~mask] = 7.787 * xyz[~mask] + 16/116
        
        lab = np.zeros_like(xyz)
        lab[..., 0] = 116 * xyz[..., 1] - 16  # L
        lab[..., 1] = 500 * (xyz[..., 0] - xyz[..., 1])  # a
        lab[..., 2] = 200 * (xyz[..., 1] - xyz[..., 2])  # b
        
        return lab
    
    def _lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        """Convert LAB to RGB color space."""
        # LAB to XYZ
        fy = (lab[..., 0] + 16) / 116
        fx = lab[..., 1] / 500 + fy
        fz = fy - lab[..., 2] / 200
        
        xyz = np.stack([fx, fy, fz], axis=-1)
        
        mask = xyz > 0.2068966
        xyz[mask] = np.power(xyz[mask], 3)
        xyz[~mask] = (xyz[~mask] - 16/116) / 7.787
        
        xyz *= np.array([0.95047, 1.0, 1.08883])
        
        # XYZ to RGB
        M_inv = np.array([
            [3.240479, -1.537150, -0.498535],
            [-0.969256, 1.875992, 0.041556],
            [0.055648, -0.204043, 1.057311]
        ])
        
        rgb = xyz @ M_inv.T
        
        # Gamma correction
        mask = rgb > 0.0031308
        rgb[mask] = 1.055 * np.power(rgb[mask], 1/2.4) - 0.055
        rgb[~mask] *= 12.92
        
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return rgb
    
    def fit(self, target_image: np.ndarray) -> "ReinhardNormalizer":
        """Fit to target image."""
        target_lab = self._rgb_to_lab(target_image)
        self.target_means = target_lab.reshape(-1, 3).mean(axis=0)
        self.target_stds = target_lab.reshape(-1, 3).std(axis=0)
        return self
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to target."""
        if self.target_means is None:
            raise ValueError("Must call fit() before transform()")
        
        # Convert to LAB
        source_lab = self._rgb_to_lab(image)
        
        # Get source statistics
        source_means = source_lab.reshape(-1, 3).mean(axis=0)
        source_stds = source_lab.reshape(-1, 3).std(axis=0)
        
        # Normalize
        normalized_lab = (source_lab - source_means) / (source_stds + 1e-10)
        normalized_lab = normalized_lab * self.target_stds + self.target_means
        
        # Convert back to RGB
        normalized_rgb = self._lab_to_rgb(normalized_lab)
        
        return normalized_rgb


def normalize_stain(
    image: np.ndarray,
    target: np.ndarray,
    method: str = "macenko"
) -> np.ndarray:
    """
    Convenience function for stain normalization.
    
    Args:
        image: Source image (H, W, 3) RGB
        target: Target image (H, W, 3) RGB
        method: "macenko" or "reinhard"
    
    Returns:
        Normalized image (H, W, 3) RGB
    """
    if method == "macenko":
        normalizer = MacenkoNormalizer()
    elif method == "reinhard":
        normalizer = ReinhardNormalizer()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return normalizer.fit_transform(target, image)
