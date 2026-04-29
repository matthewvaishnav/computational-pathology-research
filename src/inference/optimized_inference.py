#!/usr/bin/env python3
"""
Optimized Inference Engine

High-performance inference engine targeting <10s processing time.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .model_loader import ModelLoader, get_model_loader
from .optimized_preprocessing import OptimizedImagePreprocessor

logger = logging.getLogger(__name__)


class OptimizedInferenceEngine:
    """High-performance inference engine with GPU acceleration."""

    def __init__(
        self,
        model_loader: Optional[ModelLoader] = None,
        use_mixed_precision: bool = True,
        enable_tensorrt: bool = True,
        batch_size: int = 8,
    ):
        """Initialize optimized inference engine.

        Args:
            model_loader: Model loader instance
            use_mixed_precision: Enable automatic mixed precision
            enable_tensorrt: Enable TensorRT optimization (if available)
            batch_size: Default batch size for processing
        """
        self.model_loader = model_loader or get_model_loader()
        self.preprocessor = OptimizedImagePreprocessor()
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.enable_tensorrt = enable_tensorrt
        self.batch_size = batch_size

        # Initialize mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        # Model cache for warmed models
        self._model_cache = {}
        self._tensorrt_cache = {}

        logger.info(
            f"OptimizedInferenceEngine initialized: "
            f"mixed_precision={use_mixed_precision}, "
            f"tensorrt={enable_tensorrt}, "
            f"batch_size={batch_size}"
        )

    def analyze_image_fast(self, image: Image.Image, disease_type: str = "breast_cancer") -> Dict:
        """Fast single image analysis.

        Args:
            image: PIL Image to analyze
            disease_type: Disease type for model selection

        Returns:
            Analysis results with timing info
        """
        start_time = time.perf_counter()

        try:
            # Get optimized model
            model = self._get_optimized_model(disease_type)

            # Fast preprocessing
            preprocess_start = time.perf_counter()
            input_tensor = self.preprocessor.preprocess_single_fast(image)
            preprocess_time = time.perf_counter() - preprocess_start

            # Fast inference
            inference_start = time.perf_counter()
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = model(input_tensor)
                else:
                    logits = model(input_tensor)

                probabilities = F.softmax(logits, dim=1)
            inference_time = time.perf_counter() - inference_start

            # Fast postprocessing
            postprocess_start = time.perf_counter()
            result = self._postprocess_fast(probabilities, disease_type)
            postprocess_time = time.perf_counter() - postprocess_start

            total_time = time.perf_counter() - start_time

            # Add timing information
            result.update(
                {
                    "timing": {
                        "total_time": total_time,
                        "preprocess_time": preprocess_time,
                        "inference_time": inference_time,
                        "postprocess_time": postprocess_time,
                    },
                    "performance": {
                        "meets_target": total_time < 10.0,
                        "fps": 1.0 / total_time if total_time > 0 else 0,
                    },
                }
            )

            logger.info(
                f"Fast inference: {total_time:.3f}s total "
                f"(preprocess: {preprocess_time:.3f}s, "
                f"inference: {inference_time:.3f}s, "
                f"postprocess: {postprocess_time:.3f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"Fast inference failed: {e}")
            raise

    def analyze_batch_fast(
        self, images: List[Image.Image], disease_type: str = "breast_cancer"
    ) -> List[Dict]:
        """Fast batch analysis.

        Args:
            images: List of PIL Images
            disease_type: Disease type for model selection

        Returns:
            List of analysis results
        """
        start_time = time.perf_counter()

        try:
            # Get optimized model
            model = self._get_optimized_model(disease_type)

            # Batch preprocessing
            preprocess_start = time.perf_counter()
            batch_tensor = self.preprocessor.preprocess_batch(images)
            preprocess_time = time.perf_counter() - preprocess_start

            # Batch inference
            inference_start = time.perf_counter()
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = model(batch_tensor)
                else:
                    logits = model(batch_tensor)

                probabilities = F.softmax(logits, dim=1)
            inference_time = time.perf_counter() - inference_start

            # Batch postprocessing
            postprocess_start = time.perf_counter()
            results = []
            for i in range(len(images)):
                prob = probabilities[i : i + 1]  # Keep batch dimension
                result = self._postprocess_fast(prob, disease_type)
                results.append(result)
            postprocess_time = time.perf_counter() - postprocess_start

            total_time = time.perf_counter() - start_time
            avg_time_per_image = total_time / len(images)

            # Add timing to all results
            timing_info = {
                "timing": {
                    "total_batch_time": total_time,
                    "avg_time_per_image": avg_time_per_image,
                    "preprocess_time": preprocess_time,
                    "inference_time": inference_time,
                    "postprocess_time": postprocess_time,
                },
                "performance": {
                    "meets_target": avg_time_per_image < 10.0,
                    "batch_fps": len(images) / total_time if total_time > 0 else 0,
                },
            }

            for result in results:
                result.update(timing_info)

            logger.info(
                f"Batch inference: {total_time:.3f}s total, "
                f"{avg_time_per_image:.3f}s avg per image, "
                f"{len(images)/total_time:.1f} FPS"
            )

            return results

        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise

    def analyze_with_tta_fast(
        self, image: Image.Image, disease_type: str = "breast_cancer"
    ) -> Dict:
        """Fast test-time augmentation analysis.

        Args:
            image: PIL Image to analyze
            disease_type: Disease type

        Returns:
            Analysis results with TTA
        """
        start_time = time.perf_counter()

        try:
            model = self._get_optimized_model(disease_type)

            # TTA preprocessing (4 augmentations)
            preprocess_start = time.perf_counter()
            tta_tensor = self.preprocessor.preprocess_with_tta_fast(image)
            preprocess_time = time.perf_counter() - preprocess_start

            # TTA inference
            inference_start = time.perf_counter()
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = model(tta_tensor)
                else:
                    logits = model(tta_tensor)

                probabilities = F.softmax(logits, dim=1)

                # Average predictions across augmentations
                avg_probabilities = probabilities.mean(dim=0, keepdim=True)
            inference_time = time.perf_counter() - inference_start

            # Postprocessing
            postprocess_start = time.perf_counter()
            result = self._postprocess_fast(avg_probabilities, disease_type)

            # Add TTA-specific info
            result["tta_info"] = {
                "num_augmentations": tta_tensor.shape[0],
                "individual_predictions": probabilities.cpu().numpy().tolist(),
                "prediction_variance": probabilities.var(dim=0).cpu().numpy().tolist(),
            }
            postprocess_time = time.perf_counter() - postprocess_start

            total_time = time.perf_counter() - start_time

            result.update(
                {
                    "timing": {
                        "total_time": total_time,
                        "preprocess_time": preprocess_time,
                        "inference_time": inference_time,
                        "postprocess_time": postprocess_time,
                    },
                    "performance": {
                        "meets_target": total_time < 10.0,
                        "fps": 1.0 / total_time if total_time > 0 else 0,
                    },
                }
            )

            logger.info(f"TTA inference: {total_time:.3f}s total")

            return result

        except Exception as e:
            logger.error(f"TTA inference failed: {e}")
            raise

    def _get_optimized_model(self, disease_type: str):
        """Get optimized model (with caching and TensorRT if available)."""
        cache_key = f"{disease_type}_optimized"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # Get base model
        model_info = self.model_loader.get_model(disease_type)
        model = model_info["model"]

        # Optimize model
        model = self._optimize_model(model, disease_type)

        # Cache optimized model
        self._model_cache[cache_key] = model

        return model

    def _optimize_model(self, model, disease_type: str):
        """Apply model optimizations."""
        # Ensure model is in eval mode
        model.eval()

        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()

        # Try TensorRT optimization
        if self.enable_tensorrt and torch.cuda.is_available():
            try:
                model = self._apply_tensorrt(model, disease_type)
                logger.info(f"TensorRT optimization applied for {disease_type}")
            except Exception as e:
                logger.warning(f"TensorRT optimization failed: {e}")

        # Compile model (PyTorch 2.0+)
        try:
            if hasattr(torch, "compile"):
                model = torch.compile(model, mode="max-autotune")
                logger.info(f"Model compiled for {disease_type}")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")

        return model

    def _apply_tensorrt(self, model, disease_type: str):
        """Apply TensorRT optimization if available."""
        try:
            import torch_tensorrt

            # Create example input
            example_input = torch.randn(1, 3, 224, 224).cuda()

            # Compile with TensorRT
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[example_input],
                enabled_precisions=(
                    {torch.float, torch.half} if self.use_mixed_precision else {torch.float}
                ),
                workspace_size=1 << 30,  # 1GB
                max_batch_size=self.batch_size,
            )

            return trt_model

        except ImportError:
            logger.warning("TensorRT not available, skipping optimization")
            return model
        except Exception as e:
            logger.warning(f"TensorRT compilation failed: {e}")
            return model

    def _postprocess_fast(self, probabilities: torch.Tensor, disease_type: str) -> Dict:
        """Fast postprocessing of model outputs."""
        # Get class names (simplified)
        class_names = ["benign", "malignant"]  # PCam classes

        # Get predictions
        probs_cpu = probabilities.cpu().numpy()[0]  # Remove batch dimension
        pred_class_idx = int(probs_cpu.argmax())
        confidence = float(probs_cpu.max())

        # Create result
        result = {
            "prediction_class": class_names[pred_class_idx],
            "confidence_score": confidence,
            "probability_scores": {
                class_names[i]: float(probs_cpu[i]) for i in range(len(class_names))
            },
            "model_info": {
                "disease_type": disease_type,
                "model_name": "optimized_pcam",
                "version": "1.0",
            },
        }

        return result

    def warm_up_models(self, disease_types: List[str] = None):
        """Warm up models for faster first inference.

        Args:
            disease_types: List of disease types to warm up
        """
        if disease_types is None:
            disease_types = ["breast_cancer"]

        logger.info(f"Warming up models: {disease_types}")

        for disease_type in disease_types:
            try:
                # Get and cache optimized model
                model = self._get_optimized_model(disease_type)

                # Run dummy inference
                dummy_input = torch.randn(1, 3, 224, 224)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()

                with torch.no_grad():
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            _ = model(dummy_input)
                    else:
                        _ = model(dummy_input)

                logger.info(f"Model warmed up: {disease_type}")

            except Exception as e:
                logger.error(f"Failed to warm up {disease_type}: {e}")

    def benchmark_performance(self, test_image: Image.Image, num_runs: int = 50) -> Dict:
        """Benchmark inference performance.

        Args:
            test_image: Test image
            num_runs: Number of benchmark runs

        Returns:
            Performance statistics
        """
        logger.info(f"Benchmarking performance with {num_runs} runs...")

        # Warm up
        for _ in range(5):
            _ = self.analyze_image_fast(test_image)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            result = self.analyze_image_fast(test_image)
            times.append(time.perf_counter() - start)

        import numpy as np

        stats = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "p95_time": np.percentile(times, 95),
            "p99_time": np.percentile(times, 99),
            "throughput_fps": 1.0 / np.mean(times),
            "meets_target": np.mean(times) < 10.0,
            "success_rate": np.mean([t < 10.0 for t in times]) * 100,
        }

        logger.info(f"Performance benchmark results:")
        logger.info(f"  Mean time: {stats['mean_time']:.3f}s")
        logger.info(f"  P95 time: {stats['p95_time']:.3f}s")
        logger.info(f"  Throughput: {stats['throughput_fps']:.1f} FPS")
        logger.info(f"  Meets <10s target: {stats['meets_target']}")
        logger.info(f"  Success rate: {stats['success_rate']:.1f}%")

        return stats

    def get_optimization_info(self) -> Dict:
        """Get information about applied optimizations."""
        return {
            "mixed_precision": self.use_mixed_precision,
            "tensorrt_enabled": self.enable_tensorrt,
            "batch_size": self.batch_size,
            "cached_models": list(self._model_cache.keys()),
            "device": str(self.preprocessor.device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }


if __name__ == "__main__":
    # Quick performance test
    logging.basicConfig(level=logging.INFO)

    from .optimized_preprocessing import create_test_image

    # Create test image
    test_image = create_test_image()

    # Create optimized engine
    engine = OptimizedInferenceEngine()

    # Warm up
    engine.warm_up_models()

    # Benchmark
    stats = engine.benchmark_performance(test_image, num_runs=20)

    print(f"\nOptimized Inference Performance:")
    print(f"  Average time: {stats['mean_time']:.3f}s")
    print(f"  Target met: {'✓' if stats['meets_target'] else '✗'}")
    print(f"  Throughput: {stats['throughput_fps']:.1f} FPS")
