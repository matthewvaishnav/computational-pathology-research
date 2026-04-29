"""
Nucleus Segmentation for Digital Pathology

Integrates StarDist for automated nucleus detection and segmentation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    STARDIST_AVAILABLE = True
except ImportError:
    STARDIST_AVAILABLE = False
    logger.warning("StarDist not available. Install with: pip install stardist")


class NucleusSegmenter:
    """
    Nucleus segmentation using StarDist pretrained models.

    Supports H&E and fluorescence images.
    """

    def __init__(
        self, model_name: str = "2D_versatile_he", prob_thresh: float = 0.5, nms_thresh: float = 0.4
    ):
        """
        Initialize nucleus segmenter.

        Args:
            model_name: StarDist model name
                - "2D_versatile_he": H&E images (default)
                - "2D_versatile_fluo": Fluorescence images
                - "2D_paper_dsb2018": DSB 2018 challenge model
            prob_thresh: Probability threshold for detection
            nms_thresh: Non-maximum suppression threshold
        """
        if not STARDIST_AVAILABLE:
            raise ImportError("StarDist not installed. Install with: pip install stardist")

        self.model_name = model_name
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.model = None

    def load_model(self):
        """Load pretrained StarDist model."""
        if self.model is None:
            logger.info(f"Loading StarDist model: {self.model_name}")
            self.model = StarDist2D.from_pretrained(self.model_name)

    def segment(self, image: np.ndarray, normalize_image: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Segment nuclei in image.

        Args:
            image: Input image (H, W, 3) or (H, W)
            normalize_image: Whether to normalize image

        Returns:
            labels: Segmentation mask (H, W) with nucleus IDs
            details: Dictionary with detection details
                - prob: Probability map
                - dist: Distance map
                - points: Nucleus centroids
        """
        self.load_model()

        # Normalize if needed
        if normalize_image:
            image = normalize(image, 1, 99.8)

        # Run prediction
        labels, details = self.model.predict_instances(
            image, prob_thresh=self.prob_thresh, nms_thresh=self.nms_thresh
        )

        return labels, details

    def segment_batch(
        self, images: List[np.ndarray], normalize_images: bool = True
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Segment nuclei in batch of images.

        Args:
            images: List of images
            normalize_images: Whether to normalize images

        Returns:
            List of (labels, details) tuples
        """
        self.load_model()

        results = []
        for image in images:
            labels, details = self.segment(image, normalize_images)
            results.append((labels, details))

        return results

    def get_nucleus_features(
        self, labels: np.ndarray, image: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Extract features from segmented nuclei.

        Args:
            labels: Segmentation mask
            image: Original image (optional, for intensity features)

        Returns:
            List of feature dictionaries per nucleus
        """
        from skimage.measure import regionprops

        props = regionprops(labels, intensity_image=image)

        features = []
        for prop in props:
            feat = {
                "label": prop.label,
                "area": prop.area,
                "perimeter": prop.perimeter,
                "eccentricity": prop.eccentricity,
                "solidity": prop.solidity,
                "centroid": prop.centroid,
                "bbox": prop.bbox,
            }

            if image is not None:
                feat["mean_intensity"] = prop.mean_intensity
                feat["max_intensity"] = prop.max_intensity
                feat["min_intensity"] = prop.min_intensity

            features.append(feat)

        return features


class TissueDetector:
    """
    Simple tissue detection for WSI processing.

    Identifies tissue regions vs background.
    """

    def __init__(self, threshold: float = 0.8, min_area: int = 1000):
        """
        Initialize tissue detector.

        Args:
            threshold: Luminosity threshold (0-1)
            min_area: Minimum tissue area in pixels
        """
        self.threshold = threshold
        self.min_area = min_area

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect tissue regions.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Binary mask (H, W) where 1 = tissue
        """
        from skimage.color import rgb2gray
        from skimage.morphology import binary_closing, disk, remove_small_objects

        # Convert to grayscale
        gray = rgb2gray(image)

        # Threshold (tissue is darker than background)
        tissue_mask = gray < self.threshold

        # Morphological operations
        tissue_mask = binary_closing(tissue_mask, disk(5))
        tissue_mask = remove_small_objects(tissue_mask, min_size=self.min_area)

        return tissue_mask.astype(np.uint8)

    def get_tissue_percentage(self, image: np.ndarray) -> float:
        """
        Calculate percentage of tissue in image.

        Args:
            image: RGB image

        Returns:
            Tissue percentage (0-100)
        """
        mask = self.detect(image)
        return 100 * mask.sum() / mask.size


def segment_nuclei(
    image: np.ndarray,
    model_name: str = "2D_versatile_he",
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.4,
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function for nucleus segmentation.

    Args:
        image: Input image
        model_name: StarDist model name
        prob_thresh: Probability threshold
        nms_thresh: NMS threshold

    Returns:
        labels: Segmentation mask
        details: Detection details
    """
    segmenter = NucleusSegmenter(model_name, prob_thresh, nms_thresh)
    return segmenter.segment(image)


def detect_tissue(image: np.ndarray, threshold: float = 0.8, min_area: int = 1000) -> np.ndarray:
    """
    Convenience function for tissue detection.

    Args:
        image: RGB image
        threshold: Luminosity threshold
        min_area: Minimum tissue area

    Returns:
        Binary tissue mask
    """
    detector = TissueDetector(threshold, min_area)
    return detector.detect(image)
