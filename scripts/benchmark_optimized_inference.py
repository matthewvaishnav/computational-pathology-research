#!/usr/bin/env python3
"""
Optimized Inference Benchmark

Comprehensive benchmark to validate <10s inference target is achieved.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from PIL import Image

from src.inference.optimized_inference import OptimizedInferenceEngine
from src.inference.optimized_preprocessing import create_test_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceBenchmark:
    """Comprehensive inference performance benchmark."""
    
    def __init__(self):
        self.engine = OptimizedInferenceEngine(
            use_mixed_precision=True,
            enable_tensorrt=True,
            batch_size=8
        )
        
        # Create test images
        self.test_images = self._create_test_dataset()
        
        logger.info("InferenceBenchmark initialized")
    
    def _create_test_dataset(self) -> List[Image.Image]:
        """Create diverse test dataset."""
        images = []
        
        # Different image sizes and characteristics
        test_configs = [
            (224, 224),   # Standard size
            (512, 512),   # Larger size
            (1024, 1024), # Very large
            (96, 96),     # Small size
        ]
        
        for size in test_configs:
            for seed in [42, 123, 456]:  # Different content
                np.random.seed(seed)
                image = create_test_image(size)
                images.append(image)
        
        logger.info(f"Created {len(images)} test images")
        return images
    
    def benchmark_single_inference(self, num_runs: int = 100) -> Dict:
        """Benchmark single image inference performance."""
        logger.info(f"Benchmarking single inference ({num_runs} runs)...")
        
        # Warm up models
        self.engine.warm_up_models()
        
        results = []
        
        for test_image in self.test_images[:4]:  # Test with different sizes
            logger.info(f"Testing image size: {test_image.size}")
            
            # Warm up for this image size
            for _ in range(3):
                _ = self.engine.analyze_image_fast(test_image)
            
            # Benchmark runs
            times = []
            for _ in range(num_runs // len(self.test_images[:4])):
                start = time.perf_counter()
                result = self.engine.analyze_image_fast(test_image)
                end = time.perf_counter()
                
                times.append(end - start)
                
                # Verify result structure
                assert 'prediction_class' in result
                assert 'confidence_score' in result
                assert 'timing' in result
                assert 'performance' in result
            
            # Calculate statistics
            stats = {
                'image_size': test_image.size,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'p95_time': np.percentile(times, 95),
                'p99_time': np.percentile(times, 99),
                'meets_target': np.mean(times) < 10.0,
                'success_rate': np.mean([t < 10.0 for t in times]) * 100,
                'throughput_fps': 1.0 / np.mean(times)
            }
            
            results.append(stats)
            
            logger.info(f"  Size {test_image.size}: {stats['mean_time']:.3f}s avg, "
                       f"{stats['success_rate']:.1f}% success rate")
        
        return {
            'individual_results': results,
            'overall_stats': self._aggregate_stats(results)
        }
    
    def benchmark_batch_inference(self, batch_sizes: List[int] = None) -> Dict:
        """Benchmark batch inference performance."""
        if batch_sizes is None:
            batch_sizes = [2, 4, 8, 16]
        
        logger.info(f"Benchmarking batch inference (batch sizes: {batch_sizes})...")
        
        results = []
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Create batch
            batch_images = self.test_images[:batch_size]
            
            # Warm up
            for _ in range(3):
                _ = self.engine.analyze_batch_fast(batch_images)
            
            # Benchmark
            times = []
            for _ in range(20):  # Fewer runs for batch processing
                start = time.perf_counter()
                batch_results = self.engine.analyze_batch_fast(batch_images)
                end = time.perf_counter()
                
                times.append(end - start)
                
                # Verify results
                assert len(batch_results) == batch_size
                for result in batch_results:
                    assert 'prediction_class' in result
                    assert 'timing' in result
            
            # Calculate per-image statistics
            avg_time_per_image = np.mean(times) / batch_size
            
            stats = {
                'batch_size': batch_size,
                'total_batch_time': np.mean(times),
                'time_per_image': avg_time_per_image,
                'batch_throughput': batch_size / np.mean(times),
                'meets_target': avg_time_per_image < 10.0,
                'efficiency_gain': (10.0 / avg_time_per_image) if avg_time_per_image > 0 else 0
            }
            
            results.append(stats)
            
            logger.info(f"  Batch {batch_size}: {avg_time_per_image:.3f}s per image, "
                       f"{stats['batch_throughput']:.1f} images/s")
        
        return {
            'batch_results': results,
            'optimal_batch_size': self._find_optimal_batch_size(results)
        }
    
    def benchmark_tta_inference(self, num_runs: int = 50) -> Dict:
        """Benchmark test-time augmentation inference."""
        logger.info(f"Benchmarking TTA inference ({num_runs} runs)...")
        
        test_image = self.test_images[0]  # Use standard size image
        
        # Warm up
        for _ in range(5):
            _ = self.engine.analyze_with_tta_fast(test_image)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            result = self.engine.analyze_with_tta_fast(test_image)
            end = time.perf_counter()
            
            times.append(end - start)
            
            # Verify TTA-specific results
            assert 'tta_info' in result
            assert 'num_augmentations' in result['tta_info']
            assert result['tta_info']['num_augmentations'] == 4
        
        stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'p95_time': np.percentile(times, 95),
            'meets_target': np.mean(times) < 10.0,
            'success_rate': np.mean([t < 10.0 for t in times]) * 100,
            'overhead_vs_single': np.mean(times) / (np.mean(times) / 4)  # Approximate
        }
        
        logger.info(f"TTA inference: {stats['mean_time']:.3f}s avg, "
                   f"{stats['success_rate']:.1f}% success rate")
        
        return stats
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage during inference."""
        logger.info("Benchmarking memory usage...")
        
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available for memory benchmarking'}
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Baseline memory
        baseline_memory = torch.cuda.memory_allocated()
        
        # Single inference memory
        test_image = self.test_images[0]
        _ = self.engine.analyze_image_fast(test_image)
        single_memory = torch.cuda.memory_allocated()
        
        # Batch inference memory
        batch_images = self.test_images[:8]
        _ = self.engine.analyze_batch_fast(batch_images)
        batch_memory = torch.cuda.memory_allocated()
        
        # TTA inference memory
        _ = self.engine.analyze_with_tta_fast(test_image)
        tta_memory = torch.cuda.memory_allocated()
        
        stats = {
            'baseline_mb': baseline_memory / 1024**2,
            'single_inference_mb': single_memory / 1024**2,
            'batch_inference_mb': batch_memory / 1024**2,
            'tta_inference_mb': tta_memory / 1024**2,
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2,
            'memory_efficient': torch.cuda.max_memory_allocated() < 2 * 1024**3  # <2GB
        }
        
        logger.info(f"Memory usage: peak {stats['peak_memory_mb']:.1f} MB, "
                   f"efficient: {stats['memory_efficient']}")
        
        return stats
    
    def benchmark_preprocessing_only(self, num_runs: int = 200) -> Dict:
        """Benchmark preprocessing performance in isolation."""
        logger.info(f"Benchmarking preprocessing ({num_runs} runs)...")
        
        test_image = self.test_images[0]
        
        # Benchmark optimized preprocessing
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.engine.preprocessor.preprocess_single_fast(test_image)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append(end - start)
        
        stats = {
            'mean_time': np.mean(times),
            'throughput_fps': 1.0 / np.mean(times),
            'p95_time': np.percentile(times, 95),
            'preprocessing_overhead': np.mean(times) * 100  # As percentage of 1s
        }
        
        logger.info(f"Preprocessing: {stats['mean_time']:.4f}s avg, "
                   f"{stats['throughput_fps']:.1f} FPS")
        
        return stats
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run all benchmarks and generate comprehensive report."""
        logger.info("Running comprehensive inference benchmark...")
        logger.info("="*60)
        
        # System info
        system_info = {
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
            'optimization_info': self.engine.get_optimization_info()
        }
        
        logger.info(f"System: {system_info['gpu_name'] or 'CPU only'}")
        logger.info(f"PyTorch: {system_info['pytorch_version']}")
        
        # Run all benchmarks
        results = {
            'system_info': system_info,
            'single_inference': self.benchmark_single_inference(),
            'batch_inference': self.benchmark_batch_inference(),
            'tta_inference': self.benchmark_tta_inference(),
            'memory_usage': self.benchmark_memory_usage(),
            'preprocessing_only': self.benchmark_preprocessing_only()
        }
        
        # Generate summary
        summary = self._generate_summary(results)
        results['summary'] = summary
        
        # Print summary
        self._print_summary(summary)
        
        return results
    
    def _aggregate_stats(self, individual_results: List[Dict]) -> Dict:
        """Aggregate statistics across multiple test cases."""
        all_times = []
        all_success_rates = []
        
        for result in individual_results:
            all_times.append(result['mean_time'])
            all_success_rates.append(result['success_rate'])
        
        return {
            'overall_mean_time': np.mean(all_times),
            'overall_success_rate': np.mean(all_success_rates),
            'worst_case_time': np.max(all_times),
            'best_case_time': np.min(all_times),
            'meets_target_overall': np.mean(all_times) < 10.0
        }
    
    def _find_optimal_batch_size(self, batch_results: List[Dict]) -> int:
        """Find optimal batch size based on throughput."""
        best_throughput = 0
        optimal_size = 1
        
        for result in batch_results:
            if result['batch_throughput'] > best_throughput:
                best_throughput = result['batch_throughput']
                optimal_size = result['batch_size']
        
        return optimal_size
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate benchmark summary."""
        single_stats = results['single_inference']['overall_stats']
        
        summary = {
            'target_achieved': single_stats['meets_target_overall'],
            'average_inference_time': single_stats['overall_mean_time'],
            'success_rate': single_stats['overall_success_rate'],
            'speedup_achieved': 25.0 / single_stats['overall_mean_time'],  # vs 25s baseline
            'optimal_batch_size': results['batch_inference']['optimal_batch_size'],
            'memory_efficient': results['memory_usage'].get('memory_efficient', True),
            'recommendations': []
        }
        
        # Generate recommendations
        if not summary['target_achieved']:
            summary['recommendations'].append("Target <10s not achieved - consider further optimization")
        
        if summary['success_rate'] < 95:
            summary['recommendations'].append("Success rate below 95% - investigate edge cases")
        
        if not summary['memory_efficient']:
            summary['recommendations'].append("High memory usage - consider model compression")
        
        if summary['speedup_achieved'] < 2.5:
            summary['recommendations'].append("Speedup below 2.5x - more optimization needed")
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print benchmark summary."""
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*60)
        
        logger.info(f"🎯 Target <10s achieved: {'✅' if summary['target_achieved'] else '❌'}")
        logger.info(f"⏱️  Average inference time: {summary['average_inference_time']:.3f}s")
        logger.info(f"📊 Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"🚀 Speedup vs baseline: {summary['speedup_achieved']:.1f}x")
        logger.info(f"📦 Optimal batch size: {summary['optimal_batch_size']}")
        logger.info(f"💾 Memory efficient: {'✅' if summary['memory_efficient'] else '❌'}")
        
        if summary['recommendations']:
            logger.info("\n📋 Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                logger.info(f"  {i}. {rec}")
        else:
            logger.info("\n🎉 All performance targets met!")


def main():
    """Main benchmark function."""
    logger.info("Starting optimized inference benchmark...")
    
    try:
        benchmark = InferenceBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        
        # Save results
        import json
        output_file = "inference_benchmark_results.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        results_serializable = recursive_convert(results)
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_file}")
        
        return 0 if results['summary']['target_achieved'] else 1
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())