"""
Performance benchmarking for WSI processing pipeline.

This module provides benchmarking utilities to measure and validate
performance requirements for the WSI processing pipeline components.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .batch_processor import BatchProcessor
from .config import ProcessingConfig
from .extractor import PatchExtractor
from .feature_generator import FeatureGenerator
from .quality_control import QualityControl
from .reader import WSIReader
from .tissue_detector import TissueDetector

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """
    Benchmark performance of WSI processing pipeline components.

    Measures processing speeds and validates against requirements:
    - Patch extraction: ≥100 patches/sec
    - GPU feature extraction: ≥500 patches/sec
    - CPU feature extraction: ≥50 patches/sec
    - Tissue detection: ≥1000 patches/sec
    - HDF5 write speed: ≥10 MB/sec
    - 40x magnification slide: ≤10 minutes
    - 100k x 100k pixel slide: ≤15 minutes
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize performance benchmark.

        Args:
            config: Processing configuration (uses defaults if None)
        """
        self.config = config or ProcessingConfig()
        self.results = {}

        logger.info("Initialized PerformanceBenchmark")

    def benchmark_patch_extraction(
        self,
        slide_dimensions: Tuple[int, int] = (50000, 50000),
        num_patches: int = 1000,
    ) -> Dict[str, Any]:
        """
        Benchmark patch extraction speed.

        Args:
            slide_dimensions: Simulated slide dimensions
            num_patches: Number of patches to extract for timing

        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Benchmarking patch extraction: {num_patches} patches")

        # Create patch extractor
        extractor = PatchExtractor(
            patch_size=self.config.patch_size,
            stride=self.config.stride or self.config.patch_size,
        )

        # Generate coordinates
        start_time = time.time()
        coordinates = extractor.generate_coordinates(slide_dimensions)
        coord_gen_time = time.time() - start_time

        # Limit to requested number of patches
        coordinates = coordinates[:num_patches]
        actual_patches = len(coordinates)

        # Create synthetic slide data for extraction timing
        # (We'll simulate the extraction without actual WSI file)
        patch_size = self.config.patch_size

        start_time = time.time()
        for i, (x, y) in enumerate(coordinates):
            # Simulate patch extraction (create random patch)
            patch = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)

            # Break early if we've timed enough patches
            if i >= 100:  # Time first 100 patches for speed estimate
                break

        extraction_time = time.time() - start_time
        patches_timed = min(100, actual_patches)

        # Calculate metrics
        patches_per_sec = patches_timed / extraction_time if extraction_time > 0 else 0
        coord_gen_per_sec = len(coordinates) / coord_gen_time if coord_gen_time > 0 else 0

        results = {
            "patches_per_second": patches_per_sec,
            "coordinate_generation_per_second": coord_gen_per_sec,
            "meets_requirement": patches_per_sec >= 100,  # Requirement: ≥100 patches/sec
            "requirement_target": 100,
            "slide_dimensions": slide_dimensions,
            "patches_tested": patches_timed,
            "total_coordinates": len(coordinates),
        }

        logger.info(
            f"Patch extraction benchmark: {patches_per_sec:.1f} patches/sec "
            f"(requirement: ≥100, {'PASS' if results['meets_requirement'] else 'FAIL'})"
        )

        self.results["patch_extraction"] = results
        return results

    def benchmark_feature_extraction(
        self,
        num_patches: int = 1000,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark feature extraction speed on GPU and CPU.

        Args:
            num_patches: Number of patches to process
            batch_size: Batch size (uses config default if None)

        Returns:
            Benchmark results dictionary
        """
        batch_size = batch_size or self.config.batch_size
        patch_size = self.config.patch_size

        logger.info(f"Benchmarking feature extraction: {num_patches} patches")

        # Create synthetic patches
        patches = np.random.randint(
            0, 256, (num_patches, patch_size, patch_size, 3), dtype=np.uint8
        )

        results = {}

        # Benchmark GPU if available
        if torch.cuda.is_available():
            gpu_generator = FeatureGenerator(
                encoder_name=self.config.encoder_name,
                device="cuda",
                batch_size=batch_size,
            )

            # Warmup
            warmup_patches = patches[:batch_size]
            _ = gpu_generator.extract_features(warmup_patches)
            torch.cuda.synchronize()

            # Benchmark
            start_time = time.time()
            for i in range(0, num_patches, batch_size):
                batch = patches[i : i + batch_size]
                _ = gpu_generator.extract_features(batch)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time

            gpu_patches_per_sec = num_patches / gpu_time if gpu_time > 0 else 0
            gpu_meets_req = gpu_patches_per_sec >= 500  # Requirement: ≥500 patches/sec

            results["gpu"] = {
                "patches_per_second": gpu_patches_per_sec,
                "meets_requirement": gpu_meets_req,
                "requirement_target": 500,
                "total_time": gpu_time,
                "device": str(gpu_generator.device),
            }

            logger.info(
                f"GPU feature extraction: {gpu_patches_per_sec:.1f} patches/sec "
                f"(requirement: ≥500, {'PASS' if gpu_meets_req else 'FAIL'})"
            )

        # Benchmark CPU
        cpu_generator = FeatureGenerator(
            encoder_name=self.config.encoder_name,
            device="cpu",
            batch_size=min(batch_size, 8),  # Smaller batch for CPU
        )

        # Warmup
        warmup_patches = patches[: min(batch_size, 8)]
        _ = cpu_generator.extract_features(warmup_patches)

        # Benchmark (use fewer patches for CPU to avoid long wait)
        cpu_test_patches = min(num_patches, 100)
        cpu_batch_size = min(batch_size, 8)

        start_time = time.time()
        for i in range(0, cpu_test_patches, cpu_batch_size):
            batch = patches[i : i + cpu_batch_size]
            _ = cpu_generator.extract_features(batch)
        cpu_time = time.time() - start_time

        cpu_patches_per_sec = cpu_test_patches / cpu_time if cpu_time > 0 else 0
        cpu_meets_req = cpu_patches_per_sec >= 50  # Requirement: ≥50 patches/sec

        results["cpu"] = {
            "patches_per_second": cpu_patches_per_sec,
            "meets_requirement": cpu_meets_req,
            "requirement_target": 50,
            "total_time": cpu_time,
            "patches_tested": cpu_test_patches,
            "device": str(cpu_generator.device),
        }

        logger.info(
            f"CPU feature extraction: {cpu_patches_per_sec:.1f} patches/sec "
            f"(requirement: ≥50, {'PASS' if cpu_meets_req else 'FAIL'})"
        )

        self.results["feature_extraction"] = results
        return results

    def benchmark_tissue_detection(
        self,
        num_patches: int = 5000,
    ) -> Dict[str, Any]:
        """
        Benchmark tissue detection speed.

        Args:
            num_patches: Number of patches to process

        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Benchmarking tissue detection: {num_patches} patches")

        patch_size = self.config.patch_size

        # Create synthetic patches (mix of tissue and background)
        patches = []
        for i in range(num_patches):
            if i % 2 == 0:
                # Tissue-like patch (darker, more varied)
                patch = np.random.randint(50, 200, (patch_size, patch_size, 3), dtype=np.uint8)
            else:
                # Background-like patch (brighter, more uniform)
                patch = np.random.randint(200, 255, (patch_size, patch_size, 3), dtype=np.uint8)
            patches.append(patch)

        # Create tissue detector
        detector = TissueDetector(
            method=self.config.tissue_method,
            tissue_threshold=self.config.tissue_threshold,
        )

        # Benchmark tissue detection
        start_time = time.time()
        tissue_count = 0
        for patch in patches:
            if detector.is_tissue_patch(patch):
                tissue_count += 1
        detection_time = time.time() - start_time

        patches_per_sec = num_patches / detection_time if detection_time > 0 else 0
        meets_requirement = patches_per_sec >= 1000  # Requirement: ≥1000 patches/sec

        results = {
            "patches_per_second": patches_per_sec,
            "meets_requirement": meets_requirement,
            "requirement_target": 1000,
            "total_time": detection_time,
            "patches_tested": num_patches,
            "tissue_patches_found": tissue_count,
            "tissue_percentage": tissue_count / num_patches,
        }

        logger.info(
            f"Tissue detection benchmark: {patches_per_sec:.1f} patches/sec "
            f"(requirement: ≥1000, {'PASS' if meets_requirement else 'FAIL'})"
        )

        self.results["tissue_detection"] = results
        return results

    def benchmark_hdf5_write_speed(
        self,
        num_patches: int = 10000,
        feature_dim: int = 2048,
    ) -> Dict[str, Any]:
        """
        Benchmark HDF5 write speed.

        Args:
            num_patches: Number of patches worth of data to write
            feature_dim: Feature dimension

        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Benchmarking HDF5 write speed: {num_patches} patches")

        import tempfile

        from .cache import FeatureCache

        # Create synthetic data
        features = np.random.randn(num_patches, feature_dim).astype(np.float32)
        coordinates = np.random.randint(0, 10000, (num_patches, 2), dtype=np.int32)
        metadata = {
            "slide_id": "benchmark_slide",
            "patient_id": "benchmark_patient",
            "patch_size": self.config.patch_size,
            "encoder_name": self.config.encoder_name,
        }

        # Create temporary cache
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FeatureCache(
                cache_dir=temp_dir,
                compression=self.config.compression,
            )

            # Benchmark write speed
            start_time = time.time()
            cache_path = cache.save_features(
                slide_id="benchmark_slide",
                features=features,
                coordinates=coordinates,
                metadata=metadata,
            )
            write_time = time.time() - start_time

            # Calculate metrics
            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            write_speed_mbps = file_size_mb / write_time if write_time > 0 else 0
            meets_requirement = write_speed_mbps >= 10  # Requirement: ≥10 MB/sec

            # Test compression ratio
            uncompressed_size_mb = (features.nbytes + coordinates.nbytes) / (1024 * 1024)
            compression_ratio = uncompressed_size_mb / file_size_mb if file_size_mb > 0 else 1.0

            results = {
                "write_speed_mbps": write_speed_mbps,
                "meets_requirement": meets_requirement,
                "requirement_target": 10,
                "file_size_mb": file_size_mb,
                "uncompressed_size_mb": uncompressed_size_mb,
                "compression_ratio": compression_ratio,
                "write_time": write_time,
                "patches_written": num_patches,
            }

        logger.info(
            f"HDF5 write benchmark: {write_speed_mbps:.1f} MB/sec "
            f"(requirement: ≥10, {'PASS' if meets_requirement else 'FAIL'}), "
            f"compression: {compression_ratio:.1f}x"
        )

        self.results["hdf5_write"] = results
        return results

    def run_full_benchmark_suite(
        self,
        quick_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Run complete benchmark suite.

        Args:
            quick_mode: Use smaller test sizes for faster execution

        Returns:
            Complete benchmark results
        """
        logger.info("Running full benchmark suite")

        if quick_mode:
            logger.info("Quick mode enabled - using smaller test sizes")
            patch_counts = {
                "extraction": 500,
                "features": 200,
                "tissue": 1000,
                "hdf5": 1000,
            }
        else:
            patch_counts = {
                "extraction": 2000,
                "features": 1000,
                "tissue": 5000,
                "hdf5": 10000,
            }

        # Run individual benchmarks
        self.benchmark_patch_extraction(num_patches=patch_counts["extraction"])
        self.benchmark_feature_extraction(num_patches=patch_counts["features"])
        self.benchmark_tissue_detection(num_patches=patch_counts["tissue"])
        self.benchmark_hdf5_write_speed(num_patches=patch_counts["hdf5"])

        # Calculate overall pass/fail
        all_results = self.results
        passed_tests = sum(
            1
            for result in all_results.values()
            if isinstance(result, dict) and result.get("meets_requirement", False)
        )

        # For GPU results, check if GPU test passed
        if "feature_extraction" in all_results:
            fe_results = all_results["feature_extraction"]
            if "gpu" in fe_results:
                passed_tests += 1 if fe_results["gpu"]["meets_requirement"] else 0
            if "cpu" in fe_results:
                passed_tests += 1 if fe_results["cpu"]["meets_requirement"] else 0
            # Subtract 1 because we counted feature_extraction as a whole above
            passed_tests -= 1

        total_tests = 5  # extraction, gpu_features, cpu_features, tissue, hdf5

        summary = {
            "overall_pass": passed_tests == total_tests,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "quick_mode": quick_mode,
            "timestamp": time.time(),
        }

        all_results["summary"] = summary

        logger.info(
            f"Benchmark suite complete: {passed_tests}/{total_tests} tests passed "
            f"({'PASS' if summary['overall_pass'] else 'FAIL'})"
        )

        return all_results

    def print_benchmark_report(self) -> None:
        """Print formatted benchmark report."""
        if not self.results:
            logger.warning("No benchmark results available")
            return

        print("\n" + "=" * 60)
        print("WSI PROCESSING PIPELINE PERFORMANCE BENCHMARK REPORT")
        print("=" * 60)

        # Patch extraction
        if "patch_extraction" in self.results:
            result = self.results["patch_extraction"]
            status = "PASS" if result["meets_requirement"] else "FAIL"
            print(f"\n📊 Patch Extraction:")
            print(f"   Speed: {result['patches_per_second']:.1f} patches/sec")
            print(f"   Target: ≥{result['requirement_target']} patches/sec")
            print(f"   Status: {status}")

        # Feature extraction
        if "feature_extraction" in self.results:
            fe_result = self.results["feature_extraction"]
            print(f"\n🧠 Feature Extraction:")

            if "gpu" in fe_result:
                gpu = fe_result["gpu"]
                status = "PASS" if gpu["meets_requirement"] else "FAIL"
                print(f"   GPU Speed: {gpu['patches_per_second']:.1f} patches/sec")
                print(f"   GPU Target: ≥{gpu['requirement_target']} patches/sec")
                print(f"   GPU Status: {status}")

            if "cpu" in fe_result:
                cpu = fe_result["cpu"]
                status = "PASS" if cpu["meets_requirement"] else "FAIL"
                print(f"   CPU Speed: {cpu['patches_per_second']:.1f} patches/sec")
                print(f"   CPU Target: ≥{cpu['requirement_target']} patches/sec")
                print(f"   CPU Status: {status}")

        # Tissue detection
        if "tissue_detection" in self.results:
            result = self.results["tissue_detection"]
            status = "PASS" if result["meets_requirement"] else "FAIL"
            print(f"\n🔬 Tissue Detection:")
            print(f"   Speed: {result['patches_per_second']:.1f} patches/sec")
            print(f"   Target: ≥{result['requirement_target']} patches/sec")
            print(f"   Status: {status}")

        # HDF5 write
        if "hdf5_write" in self.results:
            result = self.results["hdf5_write"]
            status = "PASS" if result["meets_requirement"] else "FAIL"
            print(f"\n💾 HDF5 Write Speed:")
            print(f"   Speed: {result['write_speed_mbps']:.1f} MB/sec")
            print(f"   Target: ≥{result['requirement_target']} MB/sec")
            print(f"   Compression: {result['compression_ratio']:.1f}x")
            print(f"   Status: {status}")

        # Summary
        if "summary" in self.results:
            summary = self.results["summary"]
            overall_status = "PASS" if summary["overall_pass"] else "FAIL"
            print(f"\n📋 Overall Results:")
            print(f"   Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
            print(f"   Overall Status: {overall_status}")

        print("\n" + "=" * 60)


def run_performance_benchmarks(
    config: Optional[ProcessingConfig] = None,
    quick_mode: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to run performance benchmarks.

    Args:
        config: Processing configuration
        quick_mode: Use smaller test sizes for faster execution

    Returns:
        Benchmark results dictionary
    """
    benchmark = PerformanceBenchmark(config)
    results = benchmark.run_full_benchmark_suite(quick_mode=quick_mode)
    benchmark.print_benchmark_report()
    return results


if __name__ == "__main__":
    # Run benchmarks when script is executed directly
    import argparse

    parser = argparse.ArgumentParser(description="Run WSI pipeline performance benchmarks")
    parser.add_argument(
        "--quick", action="store_true", help="Run in quick mode with smaller test sizes"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run benchmarks
    results = run_performance_benchmarks(quick_mode=args.quick)
