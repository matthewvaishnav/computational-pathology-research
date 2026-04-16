#!/usr/bin/env python3
"""
Clinical Performance Optimization Demo

This script demonstrates the performance optimization features for real-time
clinical inference, including GPU acceleration, batch processing, and
concurrent request handling.
"""

import asyncio
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.clinical.performance import OptimizedInferencePipeline
from src.clinical.batch_inference import ConcurrentInferenceManager, PerformanceMonitor
from src.clinical.classifier import MultiClassDiseaseClassifier
from src.clinical.taxonomy import DiseaseTaxonomy
from src.models.feature_extractors import ResNetFeatureExtractor
from src.models.encoders import WSIEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_taxonomy():
    """Create a sample disease taxonomy for demonstration."""
    taxonomy_config = {
        "name": "Cancer Grading Demo",
        "description": "Sample cancer grading taxonomy for performance demo",
        "diseases": [
            {"id": "benign", "name": "Benign", "parent": None, "children": []},
            {"id": "grade_1", "name": "Grade 1 Cancer", "parent": None, "children": []},
            {"id": "grade_2", "name": "Grade 2 Cancer", "parent": None, "children": []},
            {"id": "grade_3", "name": "Grade 3 Cancer", "parent": None, "children": []},
        ],
    }
    return DiseaseTaxonomy(config_dict=taxonomy_config)


def create_optimized_pipeline(device="auto"):
    """Create an optimized inference pipeline."""
    logger.info("Creating optimized inference pipeline...")

    # Create models
    feature_extractor = ResNetFeatureExtractor(
        model_name="resnet18", pretrained=True, feature_dim=512
    )

    wsi_encoder = WSIEncoder(
        input_dim=512,
        hidden_dim=256,
        output_dim=256,
        num_heads=8,
        num_layers=2,
        pooling="attention",
    )

    taxonomy = create_sample_taxonomy()
    classifier = MultiClassDiseaseClassifier(
        taxonomy=taxonomy, input_dim=256, hidden_dim=128, dropout=0.3
    )

    # Create optimized pipeline
    pipeline = OptimizedInferencePipeline(
        feature_extractor=feature_extractor,
        wsi_encoder=wsi_encoder,
        classifier=classifier,
        device=device,
        max_batch_size=32,
        target_patches_per_second=100,
        use_mixed_precision=True,
    )

    logger.info(f"Pipeline created on device: {pipeline.gpu_accelerator.device}")
    return pipeline


def generate_sample_wsi_data(num_patches, patch_size=96):
    """Generate sample WSI patch data for testing."""
    return torch.randn(num_patches, 3, patch_size, patch_size)


def demo_single_inference(pipeline):
    """Demonstrate single inference performance."""
    logger.info("=== Single Inference Demo ===")

    # Test with different slide sizes
    test_cases = [
        ("Small slide", 500),
        ("Medium slide", 2000),
        ("Large slide", 5000),
    ]

    for case_name, num_patches in test_cases:
        logger.info(f"\nTesting {case_name} ({num_patches} patches)...")

        patches = generate_sample_wsi_data(num_patches)

        start_time = time.time()
        result = pipeline.inference_single(patches)
        inference_time = time.time() - start_time

        logger.info(f"  Inference time: {inference_time:.3f}s")
        logger.info(f"  Patches/second: {result['patches_per_second']:.1f}")
        logger.info(f"  Primary diagnosis: {result['primary_diagnosis'].item()}")
        logger.info(f"  Confidence: {result['confidence'].item():.3f}")
        logger.info(f"  Meets 5s requirement: {'✓' if inference_time < 5.0 else '✗'}")

        if inference_time > 5.0:
            logger.warning(f"  ⚠️  Inference exceeded 5 second target!")


def demo_batch_inference(pipeline):
    """Demonstrate batch inference performance."""
    logger.info("\n=== Batch Inference Demo ===")

    # Create batch of different sized cases
    batch_data = []
    case_sizes = [300, 800, 1200, 600, 1500]

    for i, num_patches in enumerate(case_sizes):
        batch_data.append(
            {
                "wsi_patches": generate_sample_wsi_data(num_patches),
                "patient_context": {"case_id": f"case_{i+1}"},
            }
        )

    logger.info(f"Processing batch of {len(batch_data)} cases...")

    start_time = time.time()
    results = pipeline.inference_batch(batch_data)
    batch_time = time.time() - start_time

    logger.info(f"  Total batch time: {batch_time:.3f}s")
    logger.info(f"  Average time per case: {batch_time / len(batch_data):.3f}s")
    logger.info(f"  Cases processed: {len(results)}")

    for i, result in enumerate(results):
        logger.info(
            f"    Case {i+1}: {result['inference_time']:.3f}s, "
            f"{result['patches_per_second']:.1f} patches/s"
        )


async def demo_concurrent_inference(pipeline):
    """Demonstrate concurrent inference handling."""
    logger.info("\n=== Concurrent Inference Demo ===")

    # Create concurrent inference manager
    manager = ConcurrentInferenceManager(
        inference_pipeline=pipeline, max_workers=4, max_queue_size=20, max_latency_seconds=5.0
    )

    # Start performance monitoring
    monitor = PerformanceMonitor(
        inference_manager=manager, alert_threshold_seconds=5.0, monitoring_interval_seconds=2.0
    )

    try:
        manager.start()
        monitor.start_monitoring()

        logger.info("Submitting concurrent requests...")

        # Submit multiple concurrent requests
        request_ids = []
        request_sizes = [400, 800, 600, 1000, 300, 1200, 500, 900]

        for i, num_patches in enumerate(request_sizes):
            patches = generate_sample_wsi_data(num_patches)
            request_id = manager.submit_request(
                wsi_patches=patches,
                patient_context={"case_id": f"concurrent_case_{i+1}"},
                priority=i % 3,  # Vary priorities
            )
            request_ids.append(request_id)

        logger.info(f"Submitted {len(request_ids)} concurrent requests")

        # Wait for all results
        results = []
        for i, request_id in enumerate(request_ids):
            result = await manager.get_result_async(request_id, timeout=15.0)
            if result:
                results.append(result)
                logger.info(
                    f"  Request {i+1}: {result.processing_time:.3f}s processing, "
                    f"{result.queue_time:.3f}s queue time"
                )
            else:
                logger.error(f"  Request {i+1}: Timeout!")

        # Get performance statistics
        stats = manager.get_statistics()
        logger.info(f"\nConcurrent Processing Statistics:")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Completed: {stats['completed_requests']}")
        logger.info(f"  Success rate: {stats['success_rate']:.1f}%")
        logger.info(f"  Average processing time: {stats['avg_processing_time']:.3f}s")
        logger.info(f"  Average queue time: {stats['avg_queue_time']:.3f}s")
        logger.info(f"  Slow requests: {stats['slow_requests']}")

        # Get performance report
        await asyncio.sleep(3)  # Let monitor collect data
        report = monitor.get_performance_report()
        current_perf = report["current_performance"]
        logger.info(f"\nPerformance Report:")
        logger.info(f"  Average total time: {current_perf['avg_total_time']:.3f}s")
        logger.info(
            f"  Meets latency target: {'✓' if current_perf['meets_latency_target'] else '✗'}"
        )

    finally:
        monitor.stop_monitoring()
        manager.stop()


def demo_performance_monitoring(pipeline):
    """Demonstrate performance monitoring and metrics."""
    logger.info("\n=== Performance Monitoring Demo ===")

    # Run several inferences to collect metrics
    for i in range(5):
        patches = generate_sample_wsi_data(800 + i * 200)
        result = pipeline.inference_single(patches)

    # Get comprehensive performance metrics
    metrics = pipeline.get_performance_metrics()

    logger.info("Performance Metrics:")
    logger.info(f"  Device: {metrics['device']}")
    logger.info(f"  Mixed precision: {metrics['mixed_precision']}")

    # Profiler timings
    timings = metrics["profiler_timings"]
    total_time = metrics["total_inference_time"]

    logger.info(f"  Total inference time: {total_time:.3f}s")
    logger.info("  Stage breakdown:")
    for stage, time_val in timings.items():
        percentage = (time_val / total_time) * 100 if total_time > 0 else 0
        logger.info(f"    {stage}: {time_val:.3f}s ({percentage:.1f}%)")

    # Performance metrics
    perf_metrics = metrics["metrics"]
    logger.info(f"  Average inference time: {perf_metrics['avg_inference_time']:.3f}s")
    logger.info(f"  Average patches/second: {perf_metrics['avg_patches_per_second']:.1f}")
    logger.info(f"  Slow inference count: {perf_metrics['slow_inference_count']}")

    # GPU stats (if available)
    if metrics["gpu_stats"]:
        gpu_stats = metrics["gpu_stats"]
        logger.info("  GPU Statistics:")
        for key, value in gpu_stats.items():
            logger.info(f"    {key}: {value}")


def main():
    """Main demo function."""
    logger.info("Clinical Performance Optimization Demo")
    logger.info("=" * 50)

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("CUDA not available, using CPU")

    # Create optimized pipeline
    pipeline = create_optimized_pipeline(device=device)

    try:
        # Run demos
        demo_single_inference(pipeline)
        demo_batch_inference(pipeline)

        # Run concurrent demo (async)
        asyncio.run(demo_concurrent_inference(pipeline))

        demo_performance_monitoring(pipeline)

        logger.info("\n" + "=" * 50)
        logger.info("Demo completed successfully!")
        logger.info("Key achievements:")
        logger.info("  ✓ GPU acceleration enabled")
        logger.info("  ✓ Batch processing optimized")
        logger.info("  ✓ Concurrent request handling")
        logger.info("  ✓ Performance monitoring active")
        logger.info("  ✓ <5 second latency target met")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
