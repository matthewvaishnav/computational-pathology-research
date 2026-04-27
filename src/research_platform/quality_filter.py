"""
Quality Filtering Tools

Filter low-quality samples from datasets.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class QualityMetrics:
    """Quality metrics for sample"""
    blur_score: float
    brightness: float
    contrast: float
    saturation: float
    artifact_score: float
    overall_score: float
    passed: bool


@dataclass
class FilterConfig:
    """Quality filter config"""
    min_blur_score: float = 0.3
    min_brightness: float = 0.2
    max_brightness: float = 0.8
    min_contrast: float = 0.1
    min_saturation: float = 0.1
    max_artifact_score: float = 0.5
    min_overall_score: float = 0.5


class QualityFilter:
    """
    Quality filtering for medical images
    
    Filters:
    - Blur detection (Laplacian variance)
    - Brightness check
    - Contrast check
    - Saturation check
    - Artifact detection
    """
    
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        self.logger = logging.getLogger(__name__)
    
    def filter_image(self, image: np.ndarray) -> QualityMetrics:
        """
        Filter single image
        
        Args:
            image: Image array (H, W, C)
            
        Returns:
            Quality metrics
        """
        
        # Compute metrics
        blur_score = self._compute_blur(image)
        brightness = self._compute_brightness(image)
        contrast = self._compute_contrast(image)
        saturation = self._compute_saturation(image)
        artifact_score = self._compute_artifacts(image)
        
        # Overall score (weighted average)
        overall_score = (
            0.3 * blur_score +
            0.2 * self._normalize_brightness(brightness) +
            0.2 * contrast +
            0.1 * saturation +
            0.2 * (1.0 - artifact_score)
        )
        
        # Check thresholds
        passed = (
            blur_score >= self.config.min_blur_score and
            brightness >= self.config.min_brightness and
            brightness <= self.config.max_brightness and
            contrast >= self.config.min_contrast and
            saturation >= self.config.min_saturation and
            artifact_score <= self.config.max_artifact_score and
            overall_score >= self.config.min_overall_score
        )
        
        return QualityMetrics(
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            artifact_score=artifact_score,
            overall_score=overall_score,
            passed=passed
        )
    
    def filter_batch(self, images: List[np.ndarray]) -> List[QualityMetrics]:
        """Filter batch of images"""
        return [self.filter_image(img) for img in images]
    
    def _compute_blur(self, image: np.ndarray) -> float:
        """Compute blur score (Laplacian variance)"""
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Laplacian
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        
        # Convolve
        from scipy.ndimage import convolve
        filtered = convolve(gray, laplacian)
        
        # Variance
        variance = np.var(filtered)
        
        # Normalize to [0, 1]
        score = min(variance / 1000.0, 1.0)
        
        return score
    
    def _compute_brightness(self, image: np.ndarray) -> float:
        """Compute brightness"""
        return np.mean(image) / 255.0
    
    def _compute_contrast(self, image: np.ndarray) -> float:
        """Compute contrast (std dev)"""
        return np.std(image) / 255.0
    
    def _compute_saturation(self, image: np.ndarray) -> float:
        """Compute saturation"""
        
        if len(image.shape) != 3:
            return 0.0
        
        # RGB to HSV
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        
        delta = max_val - min_val
        
        # Saturation
        saturation = np.where(max_val > 0, delta / max_val, 0)
        
        return np.mean(saturation)
    
    def _compute_artifacts(self, image: np.ndarray) -> float:
        """Compute artifact score"""
        
        # Simple artifact detection: high-frequency noise
        from scipy.ndimage import gaussian_filter
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Smooth
        smoothed = gaussian_filter(gray, sigma=2.0)
        
        # High-frequency component
        high_freq = np.abs(gray - smoothed)
        
        # Artifact score
        score = np.mean(high_freq) / 255.0
        
        return score
    
    def _normalize_brightness(self, brightness: float) -> float:
        """Normalize brightness to [0, 1] score"""
        
        # Optimal brightness around 0.5
        optimal = 0.5
        distance = abs(brightness - optimal)
        
        # Score decreases with distance
        score = 1.0 - (distance / 0.5)
        
        return max(0.0, score)


class DatasetQualityFilter:
    """Filter entire datasets"""
    
    def __init__(self, config: FilterConfig = None):
        self.filter = QualityFilter(config)
        self.logger = logging.getLogger(__name__)
    
    def filter_dataset(self, image_paths: List[Path],
                      output_dir: Path = None) -> Tuple[List[Path], List[Path]]:
        """
        Filter dataset
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory for passed images
            
        Returns:
            (passed_paths, failed_paths)
        """
        
        passed = []
        failed = []
        
        for img_path in image_paths:
            try:
                # Load image
                image = self._load_image(img_path)
                
                # Filter
                metrics = self.filter.filter_image(image)
                
                if metrics.passed:
                    passed.append(img_path)
                    
                    # Copy to output
                    if output_dir:
                        self._copy_image(img_path, output_dir)
                else:
                    failed.append(img_path)
                    self.logger.debug(f"Failed: {img_path.name} (score={metrics.overall_score:.2f})")
                
            except Exception as e:
                self.logger.error(f"Error filtering {img_path}: {e}")
                failed.append(img_path)
        
        self.logger.info(f"Filtered: {len(passed)} passed, {len(failed)} failed")
        
        return passed, failed
    
    def get_quality_report(self, image_paths: List[Path]) -> Dict:
        """Generate quality report"""
        
        metrics_list = []
        
        for img_path in image_paths:
            try:
                image = self._load_image(img_path)
                metrics = self.filter.filter_image(image)
                metrics_list.append(metrics)
            except Exception as e:
                self.logger.error(f"Error: {e}")
        
        # Aggregate stats
        report = {
            'total_samples': len(image_paths),
            'passed': sum(1 for m in metrics_list if m.passed),
            'failed': sum(1 for m in metrics_list if not m.passed),
            'avg_blur_score': np.mean([m.blur_score for m in metrics_list]),
            'avg_brightness': np.mean([m.brightness for m in metrics_list]),
            'avg_contrast': np.mean([m.contrast for m in metrics_list]),
            'avg_saturation': np.mean([m.saturation for m in metrics_list]),
            'avg_artifact_score': np.mean([m.artifact_score for m in metrics_list]),
            'avg_overall_score': np.mean([m.overall_score for m in metrics_list])
        }
        
        return report
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image"""
        from PIL import Image
        img = Image.open(path)
        return np.array(img)
    
    def _copy_image(self, src: Path, dst_dir: Path):
        """Copy image to output"""
        import shutil
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)


# Convenience functions

def filter_images(image_paths: List[Path],
                 config: FilterConfig = None) -> Tuple[List[Path], List[Path]]:
    """Filter images by quality"""
    
    filter_obj = DatasetQualityFilter(config)
    return filter_obj.filter_dataset(image_paths)


def generate_quality_report(image_paths: List[Path],
                           config: FilterConfig = None) -> Dict:
    """Generate quality report"""
    
    filter_obj = DatasetQualityFilter(config)
    return filter_obj.get_quality_report(image_paths)
