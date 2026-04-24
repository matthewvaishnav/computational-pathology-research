"""
Quality Control for WSI processing.

This module provides quality control functionality to detect and report
quality issues in processed slides, including blur detection, artifact
detection, and tissue coverage validation.
"""

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .exceptions import ProcessingError

logger = logging.getLogger(__name__)


class QualityControl:
    """
    Detect and report quality issues in processed slides.

    Performs quality control checks including blur detection using Laplacian
    variance, artifact detection (pen marks, bubbles, folds), and tissue
    coverage validation.

    Args:
        blur_threshold: Minimum Laplacian variance for non-blurry patches
        min_tissue_coverage: Minimum tissue coverage fraction for slide (0.0-1.0)

    Example:
        >>> qc = QualityControl(blur_threshold=100.0, min_tissue_coverage=0.1)
        >>> blur_score = qc.calculate_blur_score(patch)
        >>> artifacts = qc.detect_artifacts(patch)
        >>> report = qc.generate_qc_report(slide_id, patches, features)
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        min_tissue_coverage: float = 0.1,
    ):
        """
        Initialize quality control checker.

        Args:
            blur_threshold: Minimum Laplacian variance for non-blurry patches.
                          Lower values indicate more blur. Typical range: 50-200.
            min_tissue_coverage: Minimum fraction of slide that should contain
                               tissue (0.0-1.0). Slides below this threshold
                               will trigger a warning.

        Raises:
            ValueError: If min_tissue_coverage is not in valid range
        """
        if not 0.0 <= min_tissue_coverage <= 1.0:
            raise ValueError(
                f"min_tissue_coverage must be between 0.0 and 1.0, " f"got {min_tissue_coverage}"
            )

        self.blur_threshold = blur_threshold
        self.min_tissue_coverage = min_tissue_coverage

        logger.debug(
            f"Initialized QualityControl: blur_threshold={blur_threshold}, "
            f"min_tissue_coverage={min_tissue_coverage}"
        )

    def calculate_blur_score(self, patch: np.ndarray) -> float:
        """
        Calculate Laplacian variance blur score for a patch.

        Uses the variance of the Laplacian operator as a measure of image
        sharpness. Higher values indicate sharper images, lower values
        indicate more blur.

        Args:
            patch: RGB patch array (height, width, 3) or grayscale (height, width)

        Returns:
            Blur score as float. Higher values = sharper image.
            Typical range: 0-1000+. Values below 100 often indicate blur.

        Example:
            >>> blur_score = qc.calculate_blur_score(patch)
            >>> if blur_score < qc.blur_threshold:
            ...     print(f"Blurry patch detected: score={blur_score:.2f}")
        """
        try:
            # Convert to grayscale if RGB
            if patch.ndim == 3 and patch.shape[2] == 3:
                gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            elif patch.ndim == 2:
                gray = patch
            else:
                raise ValueError(
                    f"Invalid patch shape: {patch.shape}. " f"Expected (H, W, 3) or (H, W)"
                )

            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()

            return float(variance)

        except Exception as e:
            logger.warning(f"Failed to calculate blur score: {e}")
            return 0.0

    def detect_artifacts(self, patch: np.ndarray) -> Dict[str, bool]:
        """
        Detect common artifacts in a patch.

        Detects:
        - Pen marks: High saturation in specific color channels
        - Bubbles: Circular regions with low intensity variation
        - Folds: Linear structures with high gradient

        Args:
            patch: RGB patch array (height, width, 3)

        Returns:
            Dictionary with artifact detection results:
            {
                'pen_marks': bool,
                'bubbles': bool,
                'folds': bool
            }

        Example:
            >>> artifacts = qc.detect_artifacts(patch)
            >>> if artifacts['pen_marks']:
            ...     print("Pen marks detected in patch")
        """
        artifacts = {
            "pen_marks": False,
            "bubbles": False,
            "folds": False,
        }

        try:
            if patch.ndim != 3 or patch.shape[2] != 3:
                logger.warning(f"Invalid patch shape for artifact detection: {patch.shape}")
                return artifacts

            # Detect pen marks: high saturation in specific channels
            artifacts["pen_marks"] = self._detect_pen_marks(patch)

            # Detect bubbles: circular regions with low variation
            artifacts["bubbles"] = self._detect_bubbles(patch)

            # Detect folds: linear structures with high gradient
            artifacts["folds"] = self._detect_folds(patch)

        except Exception as e:
            logger.warning(f"Failed to detect artifacts: {e}")

        return artifacts

    def _detect_pen_marks(self, patch: np.ndarray) -> bool:
        """
        Detect pen marks based on color saturation.

        Pen marks typically have high saturation in specific color channels
        (blue, green, or black).

        Args:
            patch: RGB patch array (height, width, 3)

        Returns:
            True if pen marks detected, False otherwise
        """
        try:
            # Convert to HSV for saturation analysis
            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]

            # High saturation pixels (potential pen marks)
            high_sat_mask = saturation > 150  # Threshold for 8-bit saturation

            # If more than 5% of pixels have high saturation, likely pen marks
            high_sat_ratio = high_sat_mask.sum() / high_sat_mask.size

            return high_sat_ratio > 0.05

        except Exception as e:
            logger.debug(f"Pen mark detection failed: {e}")
            return False

    def _detect_bubbles(self, patch: np.ndarray) -> bool:
        """
        Detect bubbles based on circular regions with low variation.

        Bubbles appear as bright circular regions with uniform intensity.

        Args:
            patch: RGB patch array (height, width, 3)

        Returns:
            True if bubbles detected, False otherwise
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

            # Detect bright regions (bubbles are typically bright)
            bright_mask = gray > 200  # Threshold for 8-bit intensity

            # If more than 20% of pixels are very bright, likely bubbles
            bright_ratio = bright_mask.sum() / bright_mask.size

            return bright_ratio > 0.20

        except Exception as e:
            logger.debug(f"Bubble detection failed: {e}")
            return False

    def _detect_folds(self, patch: np.ndarray) -> bool:
        """
        Detect tissue folds based on linear structures with high gradient.

        Folds appear as linear structures with sharp intensity changes.

        Args:
            patch: RGB patch array (height, width, 3)

        Returns:
            True if folds detected, False otherwise
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate gradient magnitude
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # High gradient regions (potential folds)
            high_grad_mask = grad_magnitude > 50  # Threshold for gradient

            # If more than 10% of pixels have high gradient, likely folds
            high_grad_ratio = high_grad_mask.sum() / high_grad_mask.size

            return high_grad_ratio > 0.10

        except Exception as e:
            logger.debug(f"Fold detection failed: {e}")
            return False

    def generate_qc_report(
        self,
        slide_id: str,
        patches: List[np.ndarray],
        features: np.ndarray,
        tissue_coverage: Optional[float] = None,
        patch_size: Optional[int] = None,
        expected_feature_dim: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate quality control report for a slide.

        Calculates various quality metrics including blur scores, artifact
        detection, tissue coverage, and dimension validation.

        Args:
            slide_id: Unique identifier for the slide
            patches: List of extracted patches
            features: Feature embeddings array (num_patches, feature_dim)
            tissue_coverage: Optional tissue coverage percentage (0.0-1.0)
            patch_size: Expected patch size for validation
            expected_feature_dim: Expected feature dimension for validation

        Returns:
            Dictionary containing quality control metrics:
            {
                'slide_id': str,
                'num_patches': int,
                'blur_scores': {
                    'mean': float,
                    'std': float,
                    'min': float,
                    'max': float,
                    'num_blurry': int,
                    'blurry_ratio': float
                },
                'artifacts': {
                    'pen_marks': int,
                    'bubbles': int,
                    'folds': int
                },
                'tissue_coverage': float or None,
                'low_tissue_warning': bool,
                'dimension_validation': {
                    'patch_dimensions_valid': bool,
                    'feature_dimensions_valid': bool
                },
                'warnings': List[str]
            }

        Example:
            >>> report = qc.generate_qc_report(
            ...     slide_id="slide_001",
            ...     patches=patches,
            ...     features=features,
            ...     tissue_coverage=0.15,
            ...     patch_size=256,
            ...     expected_feature_dim=2048
            ... )
            >>> print(f"Blurry patches: {report['blur_scores']['num_blurry']}")
        """
        warnings = []

        try:
            # Calculate blur scores for all patches
            blur_scores = [self.calculate_blur_score(patch) for patch in patches]
            blur_scores_array = np.array(blur_scores)

            num_blurry = np.sum(blur_scores_array < self.blur_threshold)
            blurry_ratio = num_blurry / len(blur_scores) if blur_scores else 0.0

            blur_metrics = {
                "mean": float(blur_scores_array.mean()) if len(blur_scores_array) > 0 else 0.0,
                "std": float(blur_scores_array.std()) if len(blur_scores_array) > 0 else 0.0,
                "min": float(blur_scores_array.min()) if len(blur_scores_array) > 0 else 0.0,
                "max": float(blur_scores_array.max()) if len(blur_scores_array) > 0 else 0.0,
                "num_blurry": int(num_blurry),
                "blurry_ratio": float(blurry_ratio),
            }

            # Detect artifacts in all patches
            artifact_counts = {
                "pen_marks": 0,
                "bubbles": 0,
                "folds": 0,
            }

            for patch in patches:
                artifacts = self.detect_artifacts(patch)
                for artifact_type, detected in artifacts.items():
                    if detected:
                        artifact_counts[artifact_type] += 1

            # Validate tissue coverage
            low_tissue_warning = False
            if tissue_coverage is not None:
                if tissue_coverage < self.min_tissue_coverage:
                    low_tissue_warning = True
                    warnings.append(
                        f"Low tissue coverage: {tissue_coverage:.1%} "
                        f"(minimum: {self.min_tissue_coverage:.1%})"
                    )

            # Validate patch dimensions
            patch_dimensions_valid = True
            if patch_size is not None and len(patches) > 0:
                for i, patch in enumerate(patches):
                    expected_shape = (patch_size, patch_size, 3)
                    if patch.shape != expected_shape:
                        patch_dimensions_valid = False
                        warnings.append(
                            f"Patch {i} has invalid dimensions: {patch.shape}, "
                            f"expected {expected_shape}"
                        )
                        break  # Only report first mismatch

            # Validate feature dimensions
            feature_dimensions_valid = True
            if expected_feature_dim is not None:
                if features.shape[1] != expected_feature_dim:
                    feature_dimensions_valid = False
                    warnings.append(
                        f"Feature dimension mismatch: {features.shape[1]}, "
                        f"expected {expected_feature_dim}"
                    )

            # Check if number of patches matches number of features
            if len(patches) != features.shape[0]:
                warnings.append(
                    f"Patch count mismatch: {len(patches)} patches, "
                    f"{features.shape[0]} features"
                )

            # Add blur warning if many patches are blurry
            if blurry_ratio > 0.5:
                warnings.append(f"High blur ratio: {blurry_ratio:.1%} of patches are blurry")

            # Add artifact warnings
            for artifact_type, count in artifact_counts.items():
                if count > len(patches) * 0.1:  # More than 10% of patches
                    warnings.append(f"High {artifact_type} count: {count}/{len(patches)} patches")

            report = {
                "slide_id": slide_id,
                "num_patches": len(patches),
                "blur_scores": blur_metrics,
                "artifacts": artifact_counts,
                "tissue_coverage": tissue_coverage,
                "low_tissue_warning": low_tissue_warning,
                "dimension_validation": {
                    "patch_dimensions_valid": patch_dimensions_valid,
                    "feature_dimensions_valid": feature_dimensions_valid,
                },
                "warnings": warnings,
            }

            # Log warnings
            if warnings:
                logger.warning(f"QC warnings for {slide_id}: {len(warnings)} issues found")
                for warning in warnings:
                    logger.warning(f"  - {warning}")
            else:
                logger.info(f"QC passed for {slide_id}: no issues found")

            return report

        except Exception as e:
            logger.error(f"Failed to generate QC report for {slide_id}: {e}")
            raise ProcessingError(f"QC report generation failed: {e}")
