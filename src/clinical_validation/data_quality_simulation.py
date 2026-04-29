"""
Data Quality Simulation System for Clinical Validation

Simulates varying data quality conditions across different hospital sites
to validate model robustness and performance under realistic conditions.
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Data quality levels for simulation"""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class QualityProfile:
    """Quality profile for a hospital site"""

    site_id: str
    overall_quality: QualityLevel
    noise_level: float  # 0.0 to 1.0
    blur_probability: float  # 0.0 to 1.0
    compression_artifacts: float  # 0.0 to 1.0
    color_shift: float  # 0.0 to 1.0
    missing_data_rate: float  # 0.0 to 1.0
    annotation_quality: float  # 0.0 to 1.0 (1.0 = perfect)
    scanner_type: str
    preprocessing_quality: float  # 0.0 to 1.0


class DataQualitySimulator:
    """Simulates various data quality issues for clinical validation"""

    def __init__(self, seed: int = 42):
        """Initialize the data quality simulator"""
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # Define quality profiles for different hospital types
        self.quality_profiles = self._create_quality_profiles()

    def _create_quality_profiles(self) -> Dict[str, QualityProfile]:
        """Create realistic quality profiles for different hospital types"""
        profiles = {
            "academic_medical_center": QualityProfile(
                site_id="academic_medical_center",
                overall_quality=QualityLevel.EXCELLENT,
                noise_level=0.05,
                blur_probability=0.02,
                compression_artifacts=0.01,
                color_shift=0.03,
                missing_data_rate=0.001,
                annotation_quality=0.95,
                scanner_type="Leica_Aperio_GT450",
                preprocessing_quality=0.98,
            ),
            "large_community_hospital": QualityProfile(
                site_id="large_community_hospital",
                overall_quality=QualityLevel.GOOD,
                noise_level=0.10,
                blur_probability=0.05,
                compression_artifacts=0.03,
                color_shift=0.08,
                missing_data_rate=0.005,
                annotation_quality=0.88,
                scanner_type="Hamamatsu_NanoZoomer",
                preprocessing_quality=0.92,
            ),
            "regional_hospital": QualityProfile(
                site_id="regional_hospital",
                overall_quality=QualityLevel.FAIR,
                noise_level=0.18,
                blur_probability=0.12,
                compression_artifacts=0.08,
                color_shift=0.15,
                missing_data_rate=0.015,
                annotation_quality=0.78,
                scanner_type="Aperio_ScanScope_CS",
                preprocessing_quality=0.82,
            ),
            "rural_hospital": QualityProfile(
                site_id="rural_hospital",
                overall_quality=QualityLevel.POOR,
                noise_level=0.25,
                blur_probability=0.20,
                compression_artifacts=0.15,
                color_shift=0.22,
                missing_data_rate=0.03,
                annotation_quality=0.65,
                scanner_type="Older_Aperio_System",
                preprocessing_quality=0.70,
            ),
            "resource_limited": QualityProfile(
                site_id="resource_limited",
                overall_quality=QualityLevel.CRITICAL,
                noise_level=0.35,
                blur_probability=0.30,
                compression_artifacts=0.25,
                color_shift=0.30,
                missing_data_rate=0.08,
                annotation_quality=0.50,
                scanner_type="Legacy_Scanner",
                preprocessing_quality=0.55,
            ),
        }
        return profiles

    def simulate_image_quality_degradation(
        self, image: Union[torch.Tensor, np.ndarray, Image.Image], profile: QualityProfile
    ) -> Union[torch.Tensor, np.ndarray, Image.Image]:
        """Apply quality degradation to an image based on site profile"""

        # Convert to PIL for easier manipulation
        if isinstance(image, torch.Tensor):
            # Assume CHW format, convert to HWC
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image_pil = Image.fromarray(
                    (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
            else:
                image_pil = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
            return_tensor = True
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(
                (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            )
            return_numpy = True
        else:
            image_pil = image.copy()
            return_pil = True

        # Apply various quality degradations
        degraded_image = image_pil

        # 1. Add noise
        if np.random.random() < profile.noise_level:
            degraded_image = self._add_noise(degraded_image, profile.noise_level)

        # 2. Add blur
        if np.random.random() < profile.blur_probability:
            degraded_image = self._add_blur(degraded_image, profile.blur_probability)

        # 3. Add compression artifacts
        if np.random.random() < profile.compression_artifacts:
            degraded_image = self._add_compression_artifacts(
                degraded_image, profile.compression_artifacts
            )

        # 4. Add color shift
        if np.random.random() < profile.color_shift:
            degraded_image = self._add_color_shift(degraded_image, profile.color_shift)

        # 5. Simulate scanner-specific artifacts
        degraded_image = self._add_scanner_artifacts(degraded_image, profile.scanner_type)

        # Convert back to original format
        if "return_tensor" in locals():
            img_array = np.array(degraded_image) / 255.0
            return torch.from_numpy(img_array).permute(2, 0, 1).float()
        elif "return_numpy" in locals():
            return np.array(degraded_image) / 255.0
        else:
            return degraded_image

    def _add_noise(self, image: Image.Image, noise_level: float) -> Image.Image:
        """Add Gaussian noise to image"""
        img_array = np.array(image)
        noise = np.random.normal(0, noise_level * 25, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    def _add_blur(self, image: Image.Image, blur_strength: float) -> Image.Image:
        """Add blur to simulate focus issues"""
        radius = blur_strength * 2.0
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _add_compression_artifacts(self, image: Image.Image, artifact_level: float) -> Image.Image:
        """Simulate JPEG compression artifacts"""
        # Simulate by saving/loading with low quality
        import io

        quality = max(10, int(100 - artifact_level * 80))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def _add_color_shift(self, image: Image.Image, shift_amount: float) -> Image.Image:
        """Add color shift to simulate staining variations"""
        # Random color enhancement
        enhancer = ImageEnhance.Color(image)
        color_factor = 1.0 + np.random.uniform(-shift_amount, shift_amount)
        color_shifted = enhancer.enhance(color_factor)

        # Random brightness shift
        enhancer = ImageEnhance.Brightness(color_shifted)
        brightness_factor = 1.0 + np.random.uniform(-shift_amount * 0.5, shift_amount * 0.5)
        return enhancer.enhance(brightness_factor)

    def _add_scanner_artifacts(self, image: Image.Image, scanner_type: str) -> Image.Image:
        """Add scanner-specific artifacts"""
        if "Legacy" in scanner_type or "Older" in scanner_type:
            # Add more severe artifacts for older scanners
            img_array = np.array(image)

            # Add periodic noise (scanner line artifacts)
            if np.random.random() < 0.3:
                lines = np.sin(np.arange(img_array.shape[0]) * 0.1) * 5
                img_array = img_array + lines[:, np.newaxis, np.newaxis]

            # Add vignetting
            if np.random.random() < 0.2:
                h, w = img_array.shape[:2]
                center_x, center_y = w // 2, h // 2
                Y, X = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                vignette = 1 - (dist_from_center / max_dist) * 0.3
                img_array = img_array * vignette[:, :, np.newaxis]

            return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

        return image

    def simulate_annotation_quality(self, annotations: Dict, profile: QualityProfile) -> Dict:
        """Simulate annotation quality issues"""
        degraded_annotations = annotations.copy()

        # Simulate missing annotations
        if np.random.random() < profile.missing_data_rate:
            # Randomly remove some annotations
            if "labels" in degraded_annotations:
                keep_mask = (
                    np.random.random(len(degraded_annotations["labels"]))
                    > profile.missing_data_rate
                )
                degraded_annotations["labels"] = [
                    label for i, label in enumerate(degraded_annotations["labels"]) if keep_mask[i]
                ]

        # Simulate annotation noise (incorrect labels)
        if "labels" in degraded_annotations:
            noise_rate = 1.0 - profile.annotation_quality
            for i, label in enumerate(degraded_annotations["labels"]):
                if np.random.random() < noise_rate:
                    # Flip label or add noise
                    if isinstance(label, (int, bool)):
                        degraded_annotations["labels"][i] = 1 - label
                    elif isinstance(label, float):
                        degraded_annotations["labels"][i] = np.clip(
                            label + np.random.normal(0, noise_rate), 0, 1
                        )

        return degraded_annotations

    def get_site_profile(self, site_id: str) -> QualityProfile:
        """Get quality profile for a specific site"""
        return self.quality_profiles.get(site_id, self.quality_profiles["regional_hospital"])

    def simulate_batch_quality(
        self, batch_data: Dict[str, torch.Tensor], site_profiles: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Apply quality simulation to a batch of data"""
        batch_size = len(site_profiles)
        degraded_batch = {}

        for key, tensor in batch_data.items():
            if key == "images" and tensor.dim() == 4:  # BCHW format
                degraded_images = []
                for i in range(batch_size):
                    profile = self.get_site_profile(site_profiles[i])
                    degraded_img = self.simulate_image_quality_degradation(tensor[i], profile)
                    degraded_images.append(degraded_img)
                degraded_batch[key] = torch.stack(degraded_images)
            else:
                degraded_batch[key] = tensor

        return degraded_batch

    def generate_quality_report(self, site_id: str) -> Dict:
        """Generate a quality assessment report for a site"""
        profile = self.get_site_profile(site_id)

        return {
            "site_id": site_id,
            "overall_quality": profile.overall_quality.value,
            "quality_metrics": {
                "noise_level": profile.noise_level,
                "blur_probability": profile.blur_probability,
                "compression_artifacts": profile.compression_artifacts,
                "color_shift": profile.color_shift,
                "missing_data_rate": profile.missing_data_rate,
                "annotation_quality": profile.annotation_quality,
                "preprocessing_quality": profile.preprocessing_quality,
            },
            "equipment": {"scanner_type": profile.scanner_type},
            "recommendations": self._generate_recommendations(profile),
        }

    def _generate_recommendations(self, profile: QualityProfile) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []

        if profile.noise_level > 0.2:
            recommendations.append("Consider scanner calibration to reduce noise")

        if profile.blur_probability > 0.15:
            recommendations.append("Review slide preparation and focusing procedures")

        if profile.compression_artifacts > 0.1:
            recommendations.append("Increase image compression quality settings")

        if profile.color_shift > 0.2:
            recommendations.append("Standardize staining protocols and reagents")

        if profile.annotation_quality < 0.8:
            recommendations.append("Implement additional pathologist training")

        if profile.preprocessing_quality < 0.8:
            recommendations.append("Upgrade image preprocessing pipeline")

        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Initialize simulator
    simulator = DataQualitySimulator(seed=42)

    # Create a dummy image for testing
    dummy_image = torch.randn(3, 224, 224)

    # Test quality degradation for different sites
    for site_id in simulator.quality_profiles.keys():
        profile = simulator.get_site_profile(site_id)
        degraded_image = simulator.simulate_image_quality_degradation(dummy_image, profile)

        print(f"Site: {site_id}")
        print(f"Quality Level: {profile.overall_quality.value}")
        print(f"Original shape: {dummy_image.shape}, Degraded shape: {degraded_image.shape}")

        # Generate quality report
        report = simulator.generate_quality_report(site_id)
        print(f"Quality Report: {report['quality_metrics']}")
        print(f"Recommendations: {len(report['recommendations'])} items")
        print("-" * 50)
