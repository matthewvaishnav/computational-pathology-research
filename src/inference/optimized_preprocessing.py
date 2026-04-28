#!/usr/bin/env python3
"""
Optimized Image Preprocessing

High-performance preprocessing pipeline optimized for <10s inference target.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import io

logger = logging.getLogger(__name__)


class OptimizedImagePreprocessor:
    """GPU-accelerated preprocessing pipeline with batch support."""
    
    def __init__(self, device: Optional[str] = None, max_workers: int = 4):
        """Initialize optimized preprocessor.
        
        Args:
            device: Target device ('cuda', 'cpu', or None for auto)
            max_workers: Number of threads for parallel processing
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.max_workers = max_workers
        
        # Pre-compile transforms on GPU
        self._setup_gpu_transforms()
        
        # Thread pool for parallel image loading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"OptimizedImagePreprocessor initialized: device={self.device}, workers={max_workers}")
    
    def _setup_gpu_transforms(self):
        """Setup GPU-accelerated transforms."""
        # Standard PCam normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        
        # GPU-based resize using interpolation
        self.resize_transform = transforms.Resize((224, 224), antialias=True)
    
    def preprocess_single_fast(self, image: Image.Image) -> torch.Tensor:
        """Fast single image preprocessing.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed tensor on GPU
        """
        # Convert to tensor (CPU)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize on CPU (faster for single images)
        image = image.resize((224, 224), Image.LANCZOS)
        
        # Convert to tensor and move to GPU
        tensor = torch.from_numpy(np.array(image)).float().to(self.device)
        tensor = tensor.permute(2, 0, 1) / 255.0  # HWC -> CHW, normalize to [0,1]
        
        # GPU normalization
        tensor = (tensor - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def preprocess_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Batch preprocessing for multiple images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Batched tensor on GPU
        """
        batch_size = len(images)
        
        # Pre-allocate tensor on GPU
        batch_tensor = torch.zeros(batch_size, 3, 224, 224, device=self.device)
        
        for i, image in enumerate(images):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize
            image = image.resize((224, 224), Image.LANCZOS)
            
            # Convert and normalize
            tensor = torch.from_numpy(np.array(image)).float().to(self.device)
            tensor = tensor.permute(2, 0, 1) / 255.0
            tensor = (tensor - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
            
            batch_tensor[i] = tensor
        
        return batch_tensor
    
    def preprocess_from_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess image directly from bytes (avoids file I/O).
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed tensor
        """
        image = Image.open(io.BytesIO(image_bytes))
        return self.preprocess_single_fast(image)
    
    def preprocess_parallel(self, image_paths: List[str]) -> torch.Tensor:
        """Parallel preprocessing of multiple image files.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Batched tensor
        """
        # Load images in parallel
        def load_image(path):
            return Image.open(path).convert('RGB')
        
        futures = [self.executor.submit(load_image, path) for path in image_paths]
        images = [future.result() for future in futures]
        
        # Batch preprocess
        return self.preprocess_batch(images)
    
    def preprocess_with_tta_fast(self, image: Image.Image) -> torch.Tensor:
        """Fast test-time augmentation preprocessing.
        
        Args:
            image: PIL Image
            
        Returns:
            Tensor with 4 augmented versions [4, 3, 224, 224]
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize once
        image = image.resize((224, 224), Image.LANCZOS)
        img_array = np.array(image)
        
        # Create augmentations using numpy (faster)
        augmentations = [
            img_array,  # Original
            np.fliplr(img_array),  # Horizontal flip
            np.flipud(img_array),  # Vertical flip
            np.rot90(img_array, k=1)  # 90-degree rotation
        ]
        
        # Convert all to tensor at once
        batch_tensor = torch.zeros(4, 3, 224, 224, device=self.device)
        
        for i, aug_array in enumerate(augmentations):
            tensor = torch.from_numpy(aug_array).float().to(self.device)
            tensor = tensor.permute(2, 0, 1) / 255.0
            tensor = (tensor - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
            batch_tensor[i] = tensor
        
        return batch_tensor
    
    def preprocess_cached(self, image: Image.Image, cache_key: str = None) -> torch.Tensor:
        """Preprocessing with simple caching.
        
        Args:
            image: PIL Image
            cache_key: Optional cache key
            
        Returns:
            Preprocessed tensor
        """
        if not hasattr(self, '_cache'):
            self._cache = {}
        
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self.preprocess_single_fast(image)
        
        if cache_key and len(self._cache) < 100:  # Simple cache size limit
            self._cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """Clear preprocessing cache."""
        if hasattr(self, '_cache'):
            self._cache.clear()
    
    def benchmark_preprocessing(self, image: Image.Image, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark preprocessing performance.
        
        Args:
            image: Test image
            num_runs: Number of benchmark runs
            
        Returns:
            Timing statistics
        """
        import time
        
        # Warm up
        for _ in range(5):
            _ = self.preprocess_single_fast(image)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.preprocess_single_fast(image)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            times.append(time.perf_counter() - start)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput_fps': 1.0 / np.mean(times)
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class PreprocessingBenchmark:
    """Benchmark preprocessing performance improvements."""
    
    def __init__(self):
        self.original = None  # Will import original preprocessor
        self.optimized = OptimizedImagePreprocessor()
    
    def compare_performance(self, test_image: Image.Image, num_runs: int = 50) -> Dict:
        """Compare original vs optimized preprocessing.
        
        Args:
            test_image: Test image
            num_runs: Number of benchmark runs
            
        Returns:
            Performance comparison results
        """
        # Benchmark optimized version
        opt_results = self.optimized.benchmark_preprocessing(test_image, num_runs)
        
        results = {
            'optimized': opt_results,
            'speedup': None,  # Will calculate if original is available
            'memory_usage': self._get_memory_usage()
        }
        
        logger.info(f"Optimized preprocessing: {opt_results['mean_time']:.4f}s avg, {opt_results['throughput_fps']:.1f} FPS")
        
        return results
    
    def _get_memory_usage(self) -> Dict:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'cached_mb': torch.cuda.memory_reserved() / 1024**2
            }
        return {'allocated_mb': 0, 'cached_mb': 0}


def create_test_image(size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """Create a test pathology-like image."""
    np.random.seed(42)
    
    # Create base tissue-like texture
    image_array = np.random.randint(180, 255, (*size, 3), dtype=np.uint8)
    
    # Add cell-like structures
    for i in range(0, size[0], 25):
        for j in range(0, size[1], 25):
            center_x, center_y = i + 12, j + 12
            radius = np.random.randint(5, 12)
            
            # Create circular structure
            y, x = np.ogrid[:size[0], :size[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Random cell color (purple/pink for pathology)
            color = np.random.choice([
                [200, 150, 200],  # Light purple
                [180, 120, 180],  # Medium purple
                [220, 180, 220],  # Pink
            ])
            
            image_array[mask] = color
    
    return Image.fromarray(image_array)


if __name__ == "__main__":
    # Quick benchmark
    logging.basicConfig(level=logging.INFO)
    
    # Create test image
    test_image = create_test_image()
    
    # Benchmark optimized preprocessing
    benchmark = PreprocessingBenchmark()
    results = benchmark.compare_performance(test_image, num_runs=100)
    
    print(f"Optimized preprocessing performance:")
    print(f"  Average time: {results['optimized']['mean_time']:.4f}s")
    print(f"  Throughput: {results['optimized']['throughput_fps']:.1f} FPS")
    print(f"  GPU memory: {results['memory_usage']['allocated_mb']:.1f} MB")