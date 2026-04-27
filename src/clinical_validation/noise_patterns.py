"""
Realistic Noise Pattern Generation for Clinical Validation

Implements various types of noise patterns commonly found in histopathology
images to test model robustness under realistic clinical conditions.
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import random
from PIL import Image, ImageFilter
from scipy import ndimage, signal
from skimage import util, filters
import logging

logger = logging.getLogger(__name__)

class NoiseType(Enum):
    """Types of noise patterns in histopathology images"""
    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    SALT_PEPPER = "salt_pepper"
    SPECKLE = "speckle"
    PERIODIC = "periodic"
    IMPULSE = "impulse"
    QUANTIZATION = "quantization"
    THERMAL = "thermal"
    SHOT = "shot"
    READOUT = "readout"

@dataclass
class NoiseProfile:
    """Noise profile configuration"""
    noise_type: NoiseType
    intensity: float  # 0.0 to 1.0
    frequency: Optional[float] = None  # For periodic noise
    correlation: Optional[float] = None  # Spatial correlation
    temporal_variation: bool = False  # Time-varying noise
    scanner_specific: bool = False  # Scanner-specific characteristics

class RealisticNoiseGenerator:
    """Generates realistic noise patterns for histopathology images"""
    
    def __init__(self, seed: int = 42):
        """Initialize noise generator"""
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        # Define scanner-specific noise characteristics
        self.scanner_noise_profiles = self._create_scanner_profiles()
        
    def _create_scanner_profiles(self) -> Dict[str, List[NoiseProfile]]:
        """Create realistic noise profiles for different scanner types"""
        profiles = {
            "Leica_Aperio_GT450": [
                NoiseProfile(NoiseType.GAUSSIAN, 0.02, correlation=0.1),
                NoiseProfile(NoiseType.SHOT, 0.01),
                NoiseProfile(NoiseType.READOUT, 0.005)
            ],
            "Hamamatsu_NanoZoomer": [
                NoiseProfile(NoiseType.GAUSSIAN, 0.03, correlation=0.15),
                NoiseProfile(NoiseType.PERIODIC, 0.01, frequency=0.05),
                NoiseProfile(NoiseType.QUANTIZATION, 0.02)
            ],
            "Aperio_ScanScope_CS": [
                NoiseProfile(NoiseType.GAUSSIAN, 0.05, correlation=0.2),
                NoiseProfile(NoiseType.SALT_PEPPER, 0.001),
                NoiseProfile(NoiseType.THERMAL, 0.03)
            ],
            "Older_Aperio_System": [
                NoiseProfile(NoiseType.GAUSSIAN, 0.08, correlation=0.3),
                NoiseProfile(NoiseType.PERIODIC, 0.04, frequency=0.1),
                NoiseProfile(NoiseType.IMPULSE, 0.002),
                NoiseProfile(NoiseType.QUANTIZATION, 0.05)
            ],
            "Legacy_Scanner": [
                NoiseProfile(NoiseType.GAUSSIAN, 0.12, correlation=0.4),
                NoiseProfile(NoiseType.PERIODIC, 0.08, frequency=0.15),
                NoiseProfile(NoiseType.SALT_PEPPER, 0.005),
                NoiseProfile(NoiseType.SPECKLE, 0.06),
                NoiseProfile(NoiseType.THERMAL, 0.08)
            ]
        }
        return profiles
    
    def add_gaussian_noise(
        self, 
        image: np.ndarray, 
        intensity: float,
        correlation: float = 0.0
    ) -> np.ndarray:
        """Add Gaussian noise with optional spatial correlation"""
        if correlation > 0:
            # Generate correlated noise using convolution
            noise = np.random.normal(0, intensity, image.shape)
            kernel_size = max(3, int(correlation * 10))
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            
            if len(image.shape) == 3:
                for c in range(image.shape[2]):
                    noise[:, :, c] = cv2.filter2D(noise[:, :, c], -1, kernel)
            else:
                noise = cv2.filter2D(noise, -1, kernel)
        else:
            noise = np.random.normal(0, intensity, image.shape)
        
        return np.clip(image + noise, 0, 1)
    
    def add_poisson_noise(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Add Poisson noise (photon shot noise)"""
        # Scale image to appropriate range for Poisson
        scaled = image * (1.0 / intensity)
        noisy = np.random.poisson(scaled) * intensity
        return np.clip(noisy, 0, 1)
    
    def add_salt_pepper_noise(
        self, 
        image: np.ndarray, 
        intensity: float
    ) -> np.ndarray:
        """Add salt and pepper noise"""
        noisy = image.copy()
        
        # Salt noise (white pixels)
        salt_mask = np.random.random(image.shape[:2]) < intensity / 2
        if len(image.shape) == 3:
            noisy[salt_mask] = [1, 1, 1]
        else:
            noisy[salt_mask] = 1
        
        # Pepper noise (black pixels)
        pepper_mask = np.random.random(image.shape[:2]) < intensity / 2
        if len(image.shape) == 3:
            noisy[pepper_mask] = [0, 0, 0]
        else:
            noisy[pepper_mask] = 0
        
        return noisy
    
    def add_speckle_noise(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Add speckle noise (multiplicative noise)"""
        noise = np.random.normal(1, intensity, image.shape)
        return np.clip(image * noise, 0, 1)
    
    def add_periodic_noise(
        self, 
        image: np.ndarray, 
        intensity: float,
        frequency: float = 0.1
    ) -> np.ndarray:
        """Add periodic noise (scanner line artifacts)"""
        h, w = image.shape[:2]
        
        # Create periodic patterns
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        # Horizontal lines
        horizontal_pattern = intensity * np.sin(2 * np.pi * frequency * Y)
        
        # Vertical lines (less common but possible)
        vertical_pattern = intensity * 0.3 * np.sin(2 * np.pi * frequency * 0.7 * X)
        
        # Combine patterns
        pattern = horizontal_pattern + vertical_pattern
        
        if len(image.shape) == 3:
            pattern = np.expand_dims(pattern, axis=2)
            pattern = np.repeat(pattern, image.shape[2], axis=2)
        
        return np.clip(image + pattern, 0, 1)
    
    def add_impulse_noise(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Add impulse noise (random bright/dark spots)"""
        noisy = image.copy()
        
        # Random locations for impulses
        impulse_mask = np.random.random(image.shape[:2]) < intensity
        
        # Random impulse values
        impulse_values = np.random.uniform(0, 1, np.sum(impulse_mask))
        
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                channel = noisy[:, :, c]
                channel[impulse_mask] = impulse_values
                noisy[:, :, c] = channel
        else:
            noisy[impulse_mask] = impulse_values
        
        return noisy
    
    def add_quantization_noise(
        self, 
        image: np.ndarray, 
        intensity: float
    ) -> np.ndarray:
        """Add quantization noise (bit depth reduction)"""
        # Reduce bit depth based on intensity
        levels = max(4, int(256 * (1 - intensity)))
        quantized = np.round(image * levels) / levels
        return np.clip(quantized, 0, 1)
    
    def add_thermal_noise(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Add thermal noise (temperature-dependent)"""
        # Thermal noise is typically Gaussian with temperature dependence
        # Simulate higher noise in "warmer" regions (brighter areas)
        brightness = np.mean(image, axis=2) if len(image.shape) == 3 else image
        thermal_factor = 1 + intensity * brightness
        
        noise = np.random.normal(0, intensity, image.shape)
        if len(image.shape) == 3:
            thermal_factor = np.expand_dims(thermal_factor, axis=2)
            thermal_factor = np.repeat(thermal_factor, image.shape[2], axis=2)
        
        thermal_noise = noise * thermal_factor
        return np.clip(image + thermal_noise, 0, 1)
    
    def add_shot_noise(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Add shot noise (photon counting noise)"""
        # Shot noise follows Poisson distribution
        # Intensity determines the "photon count" scaling
        photon_count = image / intensity
        noisy_photons = np.random.poisson(photon_count)
        return np.clip(noisy_photons * intensity, 0, 1)
    
    def add_readout_noise(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Add readout noise (CCD/CMOS sensor noise)"""
        # Readout noise is typically uniform Gaussian
        noise = np.random.normal(0, intensity, image.shape)
        return np.clip(image + noise, 0, 1)
    
    def apply_noise_profile(
        self, 
        image: np.ndarray, 
        profile: NoiseProfile
    ) -> np.ndarray:
        """Apply a specific noise profile to an image"""
        if profile.noise_type == NoiseType.GAUSSIAN:
            return self.add_gaussian_noise(
                image, profile.intensity, profile.correlation or 0.0
            )
        elif profile.noise_type == NoiseType.POISSON:
            return self.add_poisson_noise(image, profile.intensity)
        elif profile.noise_type == NoiseType.SALT_PEPPER:
            return self.add_salt_pepper_noise(image, profile.intensity)
        elif profile.noise_type == NoiseType.SPECKLE:
            return self.add_speckle_noise(image, profile.intensity)
        elif profile.noise_type == NoiseType.PERIODIC:
            return self.add_periodic_noise(
                image, profile.intensity, profile.frequency or 0.1
            )
        elif profile.noise_type == NoiseType.IMPULSE:
            return self.add_impulse_noise(image, profile.intensity)
        elif profile.noise_type == NoiseType.QUANTIZATION:
            return self.add_quantization_noise(image, profile.intensity)
        elif profile.noise_type == NoiseType.THERMAL:
            return self.add_thermal_noise(image, profile.intensity)
        elif profile.noise_type == NoiseType.SHOT:
            return self.add_shot_noise(image, profile.intensity)
        elif profile.noise_type == NoiseType.READOUT:
            return self.add_readout_noise(image, profile.intensity)
        else:
            return image
    
    def apply_scanner_noise(
        self, 
        image: np.ndarray, 
        scanner_type: str,
        severity_multiplier: float = 1.0
    ) -> np.ndarray:
        """Apply scanner-specific noise patterns"""
        if scanner_type not in self.scanner_noise_profiles:
            scanner_type = "Aperio_ScanScope_CS"  # Default
        
        noisy_image = image.copy()
        profiles = self.scanner_noise_profiles[scanner_type]
        
        for profile in profiles:
            # Scale intensity by severity multiplier
            scaled_profile = NoiseProfile(
                profile.noise_type,
                profile.intensity * severity_multiplier,
                profile.frequency,
                profile.correlation,
                profile.temporal_variation,
                profile.scanner_specific
            )
            noisy_image = self.apply_noise_profile(noisy_image, scaled_profile)
        
        return noisy_image
    
    def generate_realistic_noise_batch(
        self, 
        images: torch.Tensor,
        scanner_types: List[str],
        severity_levels: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Apply realistic noise to a batch of images"""
        batch_size = images.shape[0]
        if severity_levels is None:
            severity_levels = [1.0] * batch_size
        
        noisy_batch = []
        
        for i in range(batch_size):
            # Convert tensor to numpy
            if images[i].dim() == 3:  # CHW format
                img_np = images[i].permute(1, 2, 0).cpu().numpy()
            else:
                img_np = images[i].cpu().numpy()
            
            # Apply scanner-specific noise
            noisy_img = self.apply_scanner_noise(
                img_np, 
                scanner_types[i], 
                severity_levels[i]
            )
            
            # Convert back to tensor
            if images[i].dim() == 3:
                noisy_tensor = torch.from_numpy(noisy_img).permute(2, 0, 1)
            else:
                noisy_tensor = torch.from_numpy(noisy_img)
            
            noisy_batch.append(noisy_tensor)
        
        return torch.stack(noisy_batch)
    
    def analyze_noise_characteristics(self, image: np.ndarray) -> Dict:
        """Analyze noise characteristics in an image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Estimate noise using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Estimate SNR
        signal_power = np.mean(gray ** 2)
        noise_power = np.var(gray - filters.gaussian(gray, sigma=1))
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Detect periodic patterns using FFT
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        
        # Find peaks in frequency domain
        peaks = signal.find_peaks(fft_magnitude.flatten(), height=np.percentile(fft_magnitude, 99))[0]
        periodic_score = len(peaks) / (gray.shape[0] * gray.shape[1])
        
        return {
            'laplacian_variance': float(laplacian_var),
            'estimated_snr_db': float(snr),
            'periodic_pattern_score': float(periodic_score),
            'mean_intensity': float(np.mean(gray)),
            'std_intensity': float(np.std(gray)),
            'dynamic_range': float(np.max(gray) - np.min(gray))
        }
    
    def create_noise_benchmark(
        self, 
        clean_image: np.ndarray,
        scanner_types: List[str]
    ) -> Dict[str, Dict]:
        """Create a noise benchmark comparing different scanner types"""
        benchmark = {}
        
        for scanner_type in scanner_types:
            noisy_image = self.apply_scanner_noise(clean_image, scanner_type)
            characteristics = self.analyze_noise_characteristics(noisy_image)
            
            benchmark[scanner_type] = {
                'noise_characteristics': characteristics,
                'noise_profiles': [
                    {
                        'type': profile.noise_type.value,
                        'intensity': profile.intensity,
                        'frequency': profile.frequency,
                        'correlation': profile.correlation
                    }
                    for profile in self.scanner_noise_profiles.get(scanner_type, [])
                ]
            }
        
        return benchmark

# Example usage and testing
if __name__ == "__main__":
    # Initialize noise generator
    noise_gen = RealisticNoiseGenerator(seed=42)
    
    # Create a test image
    test_image = np.random.random((224, 224, 3))
    
    # Test different noise types
    noise_types = [
        NoiseProfile(NoiseType.GAUSSIAN, 0.05, correlation=0.1),
        NoiseProfile(NoiseType.PERIODIC, 0.03, frequency=0.08),
        NoiseProfile(NoiseType.SALT_PEPPER, 0.001),
        NoiseProfile(NoiseType.SPECKLE, 0.04)
    ]
    
    print("Testing individual noise types:")
    for profile in noise_types:
        noisy = noise_gen.apply_noise_profile(test_image, profile)
        characteristics = noise_gen.analyze_noise_characteristics(noisy)
        print(f"{profile.noise_type.value}: SNR = {characteristics['estimated_snr_db']:.2f} dB")
    
    # Test scanner-specific noise
    print("\nTesting scanner-specific noise:")
    scanner_types = list(noise_gen.scanner_noise_profiles.keys())
    
    for scanner in scanner_types:
        noisy = noise_gen.apply_scanner_noise(test_image, scanner)
        characteristics = noise_gen.analyze_noise_characteristics(noisy)
        print(f"{scanner}: SNR = {characteristics['estimated_snr_db']:.2f} dB")
    
    # Create benchmark
    benchmark = noise_gen.create_noise_benchmark(test_image, scanner_types)
    print(f"\nBenchmark created for {len(benchmark)} scanner types")