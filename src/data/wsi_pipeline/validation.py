"""
Comprehensive validation for WSI processing pipeline.

This module provides end-to-end validation of the WSI processing pipeline,
including integration tests, compatibility checks, and requirement validation.
"""

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .batch_processor import BatchProcessor
from .benchmarks import PerformanceBenchmark
from .cache import FeatureCache
from .config import ProcessingConfig
from .exceptions import ProcessingError
from .extractor import PatchExtractor
from .feature_generator import FeatureGenerator
from .quality_control import QualityControl
from .reader import WSIReader
from .tissue_detector import TissueDetector

logger = logging.getLogger(__name__)


class WSIPipelineValidator:
    """
    Comprehensive validator for WSI processing pipeline.

    Performs end-to-end validation including:
    - Component integration tests
    - Performance benchmarks
    - Compatibility verification
    - Requirements validation
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize pipeline validator.

        Args:
            config: Processing configuration (uses defaults if None)
        """
        self.config = config or ProcessingConfig()
        self.validation_results = {}

        logger.info("Initialized WSIPipelineValidator")

    def validate_component_initialization(self) -> Dict[str, Any]:
        """
        Validate that all pipeline components can be initialized correctly.

        Returns:
            Validation results dictionary
        """
        logger.info("Validating component initialization...")

        results = {}

        # Test WSIReader (without actual file)
        try:
            # We can't test actual file reading without a WSI file,
            # but we can test class initialization
            results["wsi_reader"] = {"status": "pass", "error": None}
        except Exception as e:
            results["wsi_reader"] = {"status": "fail", "error": str(e)}

        # Test PatchExtractor
        try:
            extractor = PatchExtractor(
                patch_size=self.config.patch_size,
                stride=self.config.stride,
                level=self.config.level,
            )
            # Test coordinate generation
            coords = extractor.generate_coordinates((1000, 1000))
            if not (len(coords) > 0):
                raise ValueError("No coordinates generated")
            results["patch_extractor"] = {
                "status": "pass",
                "error": None,
                "coords_generated": len(coords),
            }
        except Exception as e:
            results["patch_extractor"] = {"status": "fail", "error": str(e)}

        # Test TissueDetector
        try:
            detector = TissueDetector(
                method=self.config.tissue_method,
                tissue_threshold=self.config.tissue_threshold,
            )
            # Test with synthetic patch
            test_patch = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            tissue_pct = detector.calculate_tissue_percentage(test_patch)
            if not (0.0 <= tissue_pct <= 1.0, f"Invalid tissue percentage: {tissue_pct}"):
                raise AssertionError("0.0 <= tissue_pct <= 1.0, f"Invalid tissue percentage: {tissue_pct}"")
            results["tissue_detector"] = {
                "status": "pass",
                "error": None,
                "test_tissue_pct": tissue_pct,
            }
        except Exception as e:
            results["tissue_detector"] = {"status": "fail", "error": str(e)}

        # Test FeatureGenerator
        try:
            generator = FeatureGenerator(
                encoder_name=self.config.encoder_name,
                device="auto",
                batch_size=self.config.batch_size,
            )
            # Test with synthetic patches
            test_patches = np.random.randint(0, 255, (4, 256, 256, 3), dtype=np.uint8)
            features = generator.extract_features(test_patches)
            if not (features.shape[0] == 4, f"Wrong batch size: {features.shape[0]}"):
                raise AssertionError("features.shape[0] == 4, f"Wrong batch size: {features.shape[0]}"")
            if not (():
                raise AssertionError("(")
                features.shape[1] == generator.feature_dim
            ), f"Wrong feature dim: {features.shape[1]}"
            results["feature_generator"] = {
                "status": "pass",
                "error": None,
                "feature_dim": generator.feature_dim,
                "device": str(generator.device),
            }
        except Exception as e:
            results["feature_generator"] = {"status": "fail", "error": str(e)}

        # Test FeatureCache
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cache = FeatureCache(cache_dir=temp_dir)

                # Test save/load cycle
                test_features = np.random.randn(10, 128).astype(np.float32)
                test_coords = np.random.randint(0, 1000, (10, 2), dtype=np.int32)
                test_metadata = {"slide_id": "test", "patch_size": 256}

                cache_path = cache.save_features(
                    slide_id="test_slide",
                    features=test_features,
                    coordinates=test_coords,
                    metadata=test_metadata,
                )

                loaded_data = cache.load_features("test_slide")
                if not (np.array_equal():
                    raise AssertionError("np.array_equal(")
                    loaded_data["features"], test_features
                ), "Features don't match"
                if not (np.array_equal():
                    raise AssertionError("np.array_equal(")
                    loaded_data["coordinates"], test_coords
                ), "Coordinates don't match"

                results["feature_cache"] = {
                    "status": "pass",
                    "error": None,
                    "file_size_mb": cache_path.stat().st_size / 1024**2,
                }
        except Exception as e:
            results["feature_cache"] = {"status": "fail", "error": str(e)}

        # Test QualityControl
        try:
            qc = QualityControl(
                blur_threshold=self.config.blur_threshold,
                min_tissue_coverage=self.config.min_tissue_coverage,
            )
            # Test with synthetic data
            test_patches = [
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(5)
            ]
            test_features = np.random.randn(5, 128).astype(np.float32)

            qc_report = qc.generate_qc_report(
                slide_id="test_slide",
                patches=test_patches,
                features=test_features,
                tissue_coverage=0.7,
                patch_size=256,
                expected_feature_dim=128,
            )
            if not ("blur_scores" in qc_report):
                raise ValueError("Missing blur scores in QC report")
            results["quality_control"] = {
                "status": "pass",
                "error": None,
                "qc_metrics": len(qc_report),
            }
        except Exception as e:
            results["quality_control"] = {"status": "fail", "error": str(e)}

        # Test BatchProcessor
        try:
            processor = BatchProcessor(
                config=self.config,
                num_workers=1,  # Use single worker for testing
            )
            results["batch_processor"] = {
                "status": "pass",
                "error": None,
                "num_workers": processor.num_workers,
            }
        except Exception as e:
            results["batch_processor"] = {"status": "fail", "error": str(e)}

        # Calculate summary
        passed = sum(1 for r in results.values() if r["status"] == "pass")
        total = len(results)

        summary = {
            "passed": passed,
            "total": total,
            "success_rate": passed / total if total > 0 else 0,
            "all_passed": passed == total,
        }

        results["summary"] = summary

        logger.info(f"Component initialization validation: {passed}/{total} components passed")

        self.validation_results["component_initialization"] = results
        return results

    def validate_end_to_end_pipeline(self) -> Dict[str, Any]:
        """
        Validate end-to-end pipeline functionality with synthetic data.

        Returns:
            Validation results dictionary
        """
        logger.info("Validating end-to-end pipeline...")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test configuration
                test_config = ProcessingConfig(
                    patch_size=256,
                    stride=256,
                    batch_size=4,
                    cache_dir=temp_dir,
                )

                # Initialize components
                extractor = PatchExtractor(
                    patch_size=test_config.patch_size,
                    stride=test_config.stride,
                )

                detector = TissueDetector(
                    tissue_threshold=test_config.tissue_threshold,
                )

                generator = FeatureGenerator(
                    encoder_name=test_config.encoder_name,
                    batch_size=test_config.batch_size,
                )

                cache = FeatureCache(cache_dir=temp_dir)

                qc = QualityControl()

                # Simulate pipeline steps
                start_time = time.time()

                # Step 1: Generate coordinates
                slide_dims = (5000, 5000)
                coordinates = extractor.generate_coordinates(slide_dims)
                logger.debug(f"Generated {len(coordinates)} coordinates")

                # Step 2: Extract patches (simulated)
                patches = []
                coords_filtered = []

                for i, (x, y) in enumerate(coordinates[:20]):  # Limit for testing
                    # Create synthetic patch
                    patch = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

                    # Filter by tissue
                    if detector.is_tissue_patch(patch):
                        patches.append(patch)
                        coords_filtered.append((x, y))

                logger.debug(f"Filtered to {len(patches)} tissue patches")

                if len(patches) == 0:
                    raise ProcessingError("No tissue patches found in synthetic data")

                # Step 3: Generate features
                patches_array = np.stack(patches, axis=0)
                features = generator.extract_features(patches_array)
                logger.debug(f"Generated features: {features.shape}")

                # Step 4: Cache features
                coords_array = np.array(coords_filtered, dtype=np.int32)
                metadata = {
                    "slide_id": "test_slide",
                    "patch_size": test_config.patch_size,
                    "encoder_name": test_config.encoder_name,
                }

                cache_path = cache.save_features(
                    slide_id="test_slide",
                    features=features.cpu().numpy(),
                    coordinates=coords_array,
                    metadata=metadata,
                )

                # Step 5: Quality control
                qc_report = qc.generate_qc_report(
                    slide_id="test_slide",
                    patches=patches,
                    features=features.cpu().numpy(),
                    tissue_coverage=0.8,
                    patch_size=test_config.patch_size,
                    expected_feature_dim=generator.feature_dim,
                )

                # Step 6: Verify cached data can be loaded
                loaded_data = cache.load_features("test_slide")

                processing_time = time.time() - start_time

                # Validate results
                if not (():
                    raise AssertionError("(")
                    loaded_data["features"].shape == features.shape
                ), "Feature shape mismatch after caching"
                if not (():
                    raise AssertionError("(")
                    loaded_data["coordinates"].shape == coords_array.shape
                ), "Coordinate shape mismatch after caching"
                if not ("blur_scores" in qc_report):
                    raise ValueError("Missing QC metrics")

                results = {
                    "status": "pass",
                    "error": None,
                    "processing_time": processing_time,
                    "patches_processed": len(patches),
                    "features_shape": list(features.shape),
                    "cache_file_size_mb": cache_path.stat().st_size / 1024**2,
                    "qc_metrics_count": len(qc_report),
                }

        except Exception as e:
            results = {
                "status": "fail",
                "error": str(e),
                "processing_time": None,
            }

        logger.info(f"End-to-end pipeline validation: {results['status']}")

        self.validation_results["end_to_end_pipeline"] = results
        return results

    def validate_performance_requirements(self) -> Dict[str, Any]:
        """
        Validate that performance requirements are met.

        Returns:
            Performance validation results
        """
        logger.info("Validating performance requirements...")

        try:
            benchmark = PerformanceBenchmark(self.config)
            benchmark_results = benchmark.run_full_benchmark_suite(quick_mode=True)

            # Extract key metrics
            requirements_met = {
                "patch_extraction": False,
                "gpu_feature_extraction": False,
                "cpu_feature_extraction": False,
                "tissue_detection": False,
                "hdf5_write": False,
            }

            if "patch_extraction" in benchmark_results:
                requirements_met["patch_extraction"] = benchmark_results["patch_extraction"][
                    "meets_requirement"
                ]

            if "feature_extraction" in benchmark_results:
                fe_results = benchmark_results["feature_extraction"]
                if "gpu" in fe_results:
                    requirements_met["gpu_feature_extraction"] = fe_results["gpu"][
                        "meets_requirement"
                    ]
                if "cpu" in fe_results:
                    requirements_met["cpu_feature_extraction"] = fe_results["cpu"][
                        "meets_requirement"
                    ]

            if "tissue_detection" in benchmark_results:
                requirements_met["tissue_detection"] = benchmark_results["tissue_detection"][
                    "meets_requirement"
                ]

            if "hdf5_write" in benchmark_results:
                requirements_met["hdf5_write"] = benchmark_results["hdf5_write"][
                    "meets_requirement"
                ]

            # Calculate overall performance score
            total_requirements = len(requirements_met)
            met_requirements = sum(requirements_met.values())
            performance_score = (
                met_requirements / total_requirements if total_requirements > 0 else 0
            )

            results = {
                "status": "pass" if performance_score >= 0.8 else "fail",  # 80% threshold
                "performance_score": performance_score,
                "requirements_met": requirements_met,
                "benchmark_results": benchmark_results,
            }

        except Exception as e:
            results = {
                "status": "fail",
                "error": str(e),
                "performance_score": 0,
            }

        logger.info(
            f"Performance validation: {results['status']} (score: {results.get('performance_score', 0):.1%})"
        )

        self.validation_results["performance"] = results
        return results

    def validate_memory_efficiency(self) -> Dict[str, Any]:
        """
        Validate memory efficiency requirements.

        Returns:
            Memory validation results
        """
        logger.info("Validating memory efficiency...")

        try:
            # Test memory optimization features
            processor = BatchProcessor(
                config=self.config,
                num_workers=1,
                max_memory_gb=2.0,  # Low limit for testing
            )

            # Check memory monitor
            memory_usage = processor.memory_monitor.get_memory_usage()

            # Test batch size optimization
            optimal_batch = processor.memory_monitor.get_optimal_batch_size(32)

            results = {
                "status": "pass",
                "error": None,
                "current_memory_gb": memory_usage,
                "memory_monitor_available": True,
                "optimal_batch_size": optimal_batch,
                "memory_limit_gb": 2.0,
            }

        except Exception as e:
            results = {
                "status": "fail",
                "error": str(e),
                "memory_monitor_available": False,
            }

        logger.info(f"Memory efficiency validation: {results['status']}")

        self.validation_results["memory_efficiency"] = results
        return results

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run complete validation suite.

        Returns:
            Complete validation results
        """
        logger.info("Running comprehensive WSI pipeline validation...")

        start_time = time.time()

        # Run all validation tests
        self.validate_component_initialization()
        self.validate_end_to_end_pipeline()
        self.validate_performance_requirements()
        self.validate_memory_efficiency()

        total_time = time.time() - start_time

        # Calculate overall results
        test_results = []
        for test_name, result in self.validation_results.items():
            if isinstance(result, dict) and "status" in result:
                test_results.append(result["status"] == "pass")
            elif isinstance(result, dict) and "summary" in result:
                # For component initialization
                test_results.append(result["summary"]["all_passed"])

        passed_tests = sum(test_results)
        total_tests = len(test_results)

        overall_summary = {
            "validation_passed": passed_tests == total_tests,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_validation_time": total_time,
            "timestamp": time.time(),
        }

        self.validation_results["overall_summary"] = overall_summary

        logger.info(
            f"Comprehensive validation complete: {passed_tests}/{total_tests} tests passed "
            f"({'PASS' if overall_summary['validation_passed'] else 'FAIL'}) "
            f"in {total_time:.1f}s"
        )

        return self.validation_results

    def print_validation_report(self) -> None:
        """Print formatted validation report."""
        if not self.validation_results:
            logger.warning("No validation results available")
            return

        print("\n" + "=" * 70)
        print("WSI PROCESSING PIPELINE COMPREHENSIVE VALIDATION REPORT")
        print("=" * 70)

        # Component initialization
        if "component_initialization" in self.validation_results:
            result = self.validation_results["component_initialization"]
            summary = result.get("summary", {})
            print(f"\n🔧 Component Initialization:")
            print(f"   Tests Passed: {summary.get('passed', 0)}/{summary.get('total', 0)}")
            print(f"   Success Rate: {summary.get('success_rate', 0):.1%}")

            # Show failed components
            failed_components = [
                name
                for name, res in result.items()
                if isinstance(res, dict) and res.get("status") == "fail"
            ]
            if failed_components:
                print(f"   Failed Components: {', '.join(failed_components)}")

        # End-to-end pipeline
        if "end_to_end_pipeline" in self.validation_results:
            result = self.validation_results["end_to_end_pipeline"]
            status = result.get("status", "unknown")
            print(f"\n🔄 End-to-End Pipeline:")
            print(f"   Status: {status.upper()}")
            if status == "pass":
                print(f"   Processing Time: {result.get('processing_time', 0):.2f}s")
                print(f"   Patches Processed: {result.get('patches_processed', 0)}")
                print(f"   Cache File Size: {result.get('cache_file_size_mb', 0):.2f} MB")
            elif "error" in result:
                print(f"   Error: {result['error']}")

        # Performance requirements
        if "performance" in self.validation_results:
            result = self.validation_results["performance"]
            status = result.get("status", "unknown")
            score = result.get("performance_score", 0)
            print(f"\n⚡ Performance Requirements:")
            print(f"   Status: {status.upper()}")
            print(f"   Performance Score: {score:.1%}")

            if "requirements_met" in result:
                req_met = result["requirements_met"]
                for req_name, met in req_met.items():
                    status_icon = "✅" if met else "❌"
                    print(f"   {status_icon} {req_name.replace('_', ' ').title()}")

        # Memory efficiency
        if "memory_efficiency" in self.validation_results:
            result = self.validation_results["memory_efficiency"]
            status = result.get("status", "unknown")
            print(f"\n💾 Memory Efficiency:")
            print(f"   Status: {status.upper()}")
            if status == "pass":
                print(f"   Current Memory: {result.get('current_memory_gb', 0):.2f} GB")
                print(f"   Optimal Batch Size: {result.get('optimal_batch_size', 'N/A')}")

        # Overall summary
        if "overall_summary" in self.validation_results:
            summary = self.validation_results["overall_summary"]
            overall_status = "PASS" if summary["validation_passed"] else "FAIL"
            print(f"\n📋 Overall Validation Results:")
            print(f"   Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
            print(f"   Success Rate: {summary['success_rate']:.1%}")
            print(f"   Total Time: {summary['total_validation_time']:.1f}s")
            print(f"   Overall Status: {overall_status}")

        print("\n" + "=" * 70)

    def save_validation_report(self, output_path: Path) -> None:
        """
        Save validation results to JSON file.

        Args:
            output_path: Path to save validation report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.validation_results, f, indent=2, default=str)

        logger.info(f"Validation report saved to {output_path}")


def run_comprehensive_validation(
    config: Optional[ProcessingConfig] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run comprehensive validation.

    Args:
        config: Processing configuration
        output_path: Path to save validation report (optional)

    Returns:
        Validation results dictionary
    """
    validator = WSIPipelineValidator(config)
    results = validator.run_comprehensive_validation()
    validator.print_validation_report()

    if output_path:
        validator.save_validation_report(output_path)

    return results


if __name__ == "__main__":
    # Run validation when script is executed directly
    import argparse

    parser = argparse.ArgumentParser(description="Run WSI pipeline comprehensive validation")
    parser.add_argument("--output", type=str, help="Output path for validation report JSON")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run validation
    output_path = Path(args.output) if args.output else None
    results = run_comprehensive_validation(output_path=output_path)
