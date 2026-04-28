"""
Enhanced Performance Validation for Real-Time WSI Streaming System.

This module implements the remaining critical performance validation tasks:
- 8.2.1.1: Validate 30-second processing requirement on target hardware
- 8.2.1.2: Test memory usage bounds across various slide sizes  
- 8.2.1.3: Benchmark throughput scaling with multiple GPUs

These enhanced validations provide comprehensive testing beyond the basic
performance validation already implemented.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import psutil

from .performance_validation import PerformanceValidator, MemoryMonitor
from ..utils.config import StreamingConfig

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPerformanceResult:
    """Enhanced performance validation results."""
    test_name: str
    success_rate: float
    meets_requirements: bool
    detailed_results: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: float


class EnhancedPerformanceValidator(PerformanceValidator):
    """
    Enhanced performance validator with comprehensive testing capabilities.
    
    Extends the base PerformanceValidator with additional validation methods
    for critical performance requirements.
    """
    
    def __init__(self, config: StreamingConfig):
        """Initialize enhanced performance validator."""
        super().__init__(config)
        self.enhanced_results = []
        
    async def validate_30_second_processing_requirement(
        self,
        target_hardware: str = "RTX_4090",
        test_slides: Optional[List[str]] = None
    ) -> EnhancedPerformanceResult:
        """
        Task 8.2.1.1: Validate 30-second processing requirement on target hardware.
        
        Args:
            target_hardware: Target hardware specification
            test_slides: Optional list of test slides, generates synthetic if None
            
        Returns:
            Enhanced validation results with detailed timing metrics
        """
        logger.info(f"Validating 30-second processing requirement on {target_hardware}")
        
        # Test with various slide sizes representing real clinical scenarios
        test_scenarios = [
            {"name": "Large Clinical Slide", "size": (75000, 75000), "patches": 87890},
            {"name": "Gigapixel Research Slide", "size": (100000, 100000), "patches": 156250},
            {"name": "Ultra-High Resolution", "size": (120000, 120000), "patches": 225000}
        ]
        
        results = []
        
        for scenario in test_scenarios:
            logger.info(f"Testing scenario: {scenario['name']} ({scenario['size']})")
            
            # Generate synthetic slide if no test slides provided
            if test_slides is None:
                slide_path = self._generate_synthetic_slide(
                    scenario['size'][0], scenario['size'][1]
                )
            else:
                slide_path = test_slides[0] if test_slides else None
            
            # Run processing with timing
            start_time = time.time()
            
            try:
                processing_result = await self._process_slide_with_monitoring(slide_path)
                processing_time = time.time() - start_time
                
                meets_requirement = processing_time <= 30.0
                throughput = processing_result['patches_processed'] / processing_time
                
                scenario_result = {
                    'scenario_name': scenario['name'],
                    'slide_dimensions': scenario['size'],
                    'estimated_patches': scenario['patches'],
                    'actual_patches_processed': processing_result['patches_processed'],
                    'processing_time_seconds': processing_time,
                    'meets_30_second_requirement': meets_requirement,
                    'throughput_patches_per_second': throughput,
                    'peak_memory_gb': processing_result['peak_memory_gb'],
                    'gpu_utilization_percent': processing_result.get('gpu_utilization', 0),
                    'early_stopping_triggered': processing_result.get('early_stopping', False),
                    'final_confidence': processing_result.get('confidence', 0.0)
                }
                
                results.append(scenario_result)
                
                logger.info(f"  Processing time: {processing_time:.2f}s")
                logger.info(f"  Meets requirement: {meets_requirement}")
                logger.info(f"  Throughput: {throughput:.0f} patches/sec")
                
            except Exception as e:
                logger.error(f"Failed to process scenario {scenario['name']}: {e}")
                results.append({
                    'scenario_name': scenario['name'],
                    'error': str(e),
                    'meets_30_second_requirement': False
                })
        
        # Calculate overall metrics
        successful_tests = [r for r in results if 'error' not in r]
        overall_success_rate = len([r for r in successful_tests if r['meets_30_second_requirement']]) / len(successful_tests) if successful_tests else 0
        avg_throughput = np.mean([r['throughput_patches_per_second'] for r in successful_tests]) if successful_tests else 0
        
        meets_requirements = overall_success_rate >= 0.95  # 95% success rate required
        
        # Generate recommendations
        recommendations = []
        if not meets_requirements:
            recommendations.extend([
                "Consider upgrading to RTX 4090 or better GPU",
                "Enable FP16 precision for faster inference",
                "Implement TensorRT optimization for model acceleration",
                "Increase batch size if memory allows"
            ])
        
        if avg_throughput < 3000:
            recommendations.append("Optimize feature extraction pipeline for higher throughput")
        
        result = EnhancedPerformanceResult(
            test_name="30-Second Processing Requirement",
            success_rate=overall_success_rate,
            meets_requirements=meets_requirements,
            detailed_results=results,
            recommendations=recommendations,
            timestamp=time.time()
        )
        
        self.enhanced_results.append(result)
        
        logger.info(f"30-second processing validation completed:")
        logger.info(f"  Success rate: {overall_success_rate:.1%}")
        logger.info(f"  Average throughput: {avg_throughput:.0f} patches/sec")
        logger.info(f"  Meets requirements: {meets_requirements}")
        
        return result
    
    async def test_memory_usage_bounds_comprehensive(
        self,
        memory_limits: List[float] = [1.0, 2.0, 4.0, 8.0],
        slide_sizes: List[Tuple[int, int]] = None
    ) -> EnhancedPerformanceResult:
        """
        Task 8.2.1.2: Test memory usage bounds across various slide sizes.
        
        Args:
            memory_limits: List of memory limits to test (in GB)
            slide_sizes: List of slide dimensions to test
            
        Returns:
            Comprehensive memory usage validation results
        """
        logger.info("Testing memory usage bounds across various slide sizes")
        
        if slide_sizes is None:
            slide_sizes = [
                (25000, 25000),   # Small slide
                (50000, 50000),   # Medium slide
                (75000, 75000),   # Large slide
                (100000, 100000), # Gigapixel slide
                (150000, 150000)  # Ultra-large slide
            ]
        
        results = []
        
        for memory_limit in memory_limits:
            logger.info(f"Testing with memory limit: {memory_limit} GB")
            
            # Create config for this memory limit
            test_config = StreamingConfig(
                tile_size=1024,
                batch_size=32,
                memory_budget_gb=memory_limit,
                target_time=60.0,  # Allow more time for memory-constrained scenarios
                confidence_threshold=0.95
            )
            
            memory_results = []
            
            for slide_size in slide_sizes:
                logger.info(f"  Testing slide size: {slide_size}")
                
                try:
                    # Generate synthetic slide
                    slide_path = self._generate_synthetic_slide(slide_size[0], slide_size[1])
                    
                    # Monitor memory during processing
                    with MemoryMonitor() as monitor:
                        processing_result = await self._process_slide_with_memory_optimization(
                            slide_path, test_config
                        )
                    
                    peak_memory = monitor.get_peak_memory_gb()
                    avg_memory = monitor.get_average_memory_gb()
                    
                    memory_efficiency = self._calculate_memory_efficiency(
                        slide_size, peak_memory, processing_result['patches_processed']
                    )
                    
                    slide_result = {
                        'slide_dimensions': slide_size,
                        'estimated_patches': (slide_size[0] // 1024) * (slide_size[1] // 1024),
                        'actual_patches_processed': processing_result['patches_processed'],
                        'peak_memory_gb': peak_memory,
                        'average_memory_gb': avg_memory,
                        'memory_limit_gb': memory_limit,
                        'memory_usage_within_bounds': peak_memory <= memory_limit * 1.1,  # 10% tolerance
                        'memory_efficiency_score': memory_efficiency,
                        'processing_time_seconds': processing_result.get('processing_time', 0),
                        'batch_size_adaptations': processing_result.get('batch_size_changes', 0),
                        'oom_recoveries': processing_result.get('oom_recoveries', 0)
                    }
                    
                    memory_results.append(slide_result)
                    
                    logger.info(f"    Peak memory: {peak_memory:.2f} GB")
                    logger.info(f"    Within bounds: {slide_result['memory_usage_within_bounds']}")
                    
                except Exception as e:
                    logger.error(f"Failed to process slide {slide_size}: {e}")
                    memory_results.append({
                        'slide_dimensions': slide_size,
                        'error': str(e),
                        'memory_usage_within_bounds': False
                    })
            
            # Calculate metrics for this memory limit
            successful_tests = [r for r in memory_results if 'error' not in r]
            within_bounds_rate = len([r for r in successful_tests if r['memory_usage_within_bounds']]) / len(successful_tests) if successful_tests else 0
            avg_efficiency = np.mean([r['memory_efficiency_score'] for r in successful_tests]) if successful_tests else 0
            
            results.append({
                'memory_limit_gb': memory_limit,
                'slide_results': memory_results,
                'within_bounds_success_rate': within_bounds_rate,
                'average_memory_efficiency': avg_efficiency,
                'meets_memory_requirement': within_bounds_rate >= 0.9  # 90% success rate required
            })
        
        # Overall assessment
        overall_success = all(r['meets_memory_requirement'] for r in results)
        overall_success_rate = np.mean([r['within_bounds_success_rate'] for r in results])
        
        # Generate recommendations
        recommendations = []
        if not overall_success:
            recommendations.extend([
                "Implement more aggressive batch size reduction under memory pressure",
                "Add feature caching with compression to reduce memory usage",
                "Consider using FP16 precision to halve memory requirements",
                "Implement progressive garbage collection during processing"
            ])
        
        # Find recommended memory limit
        recommended_limit = min([r['memory_limit_gb'] for r in results if r['meets_memory_requirement']], default=8.0)
        
        result = EnhancedPerformanceResult(
            test_name="Memory Usage Bounds",
            success_rate=overall_success_rate,
            meets_requirements=overall_success,
            detailed_results=results,
            recommendations=recommendations,
            timestamp=time.time()
        )
        
        self.enhanced_results.append(result)
        
        logger.info(f"Memory usage bounds validation completed:")
        logger.info(f"  Overall success rate: {overall_success_rate:.1%}")
        logger.info(f"  Recommended memory limit: {recommended_limit} GB")
        logger.info(f"  Meets requirements: {overall_success}")
        
        return result
    
    async def benchmark_throughput_scaling_comprehensive(
        self,
        gpu_configurations: List[Dict[str, Any]] = None,
        test_slide_size: Tuple[int, int] = (75000, 75000)
    ) -> EnhancedPerformanceResult:
        """
        Task 8.2.1.3: Benchmark throughput scaling with multiple GPUs.
        
        Args:
            gpu_configurations: List of GPU configurations to test
            test_slide_size: Slide dimensions for scaling tests
            
        Returns:
            Comprehensive throughput scaling benchmark results
        """
        logger.info("Benchmarking throughput scaling with multiple GPUs")
        
        if gpu_configurations is None:
            gpu_configurations = [
                {'gpu_count': 1, 'batch_size': 32, 'memory_per_gpu': 8.0},
                {'gpu_count': 2, 'batch_size': 64, 'memory_per_gpu': 8.0},
                {'gpu_count': 4, 'batch_size': 128, 'memory_per_gpu': 8.0},
                {'gpu_count': 8, 'batch_size': 256, 'memory_per_gpu': 8.0}
            ]
        
        # Filter configurations based on available GPUs
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_configurations = [cfg for cfg in gpu_configurations if cfg['gpu_count'] <= available_gpus]
        
        if not gpu_configurations:
            logger.warning("No GPU configurations available for testing")
            return EnhancedPerformanceResult(
                test_name="Throughput Scaling",
                success_rate=0.0,
                meets_requirements=False,
                detailed_results=[{'error': 'No GPUs available for scaling test'}],
                recommendations=["Install CUDA-compatible GPUs for scaling validation"],
                timestamp=time.time()
            )
        
        results = []
        baseline_throughput = None
        
        for config in gpu_configurations:
            logger.info(f"Testing {config['gpu_count']} GPU configuration")
            
            # Create streaming config for this GPU setup
            streaming_config = StreamingConfig(
                tile_size=1024,
                batch_size=config['batch_size'],
                memory_budget_gb=config['memory_per_gpu'] * config['gpu_count'],
                target_time=60.0,
                confidence_threshold=0.95,
                gpu_ids=list(range(config['gpu_count']))
            )
            
            try:
                # Generate test slide
                slide_path = self._generate_synthetic_slide(test_slide_size[0], test_slide_size[1])
                
                # Run multiple iterations for statistical significance
                throughputs = []
                processing_times = []
                
                for iteration in range(3):  # 3 iterations per configuration
                    logger.info(f"  Iteration {iteration + 1}/3")
                    
                    start_time = time.time()
                    processing_result = await self._process_slide_multi_gpu(slide_path, streaming_config)
                    processing_time = time.time() - start_time
                    
                    throughput = processing_result['patches_processed'] / processing_time
                    throughputs.append(throughput)
                    processing_times.append(processing_time)
                
                # Calculate statistics
                avg_throughput = np.mean(throughputs)
                std_throughput = np.std(throughputs)
                avg_processing_time = np.mean(processing_times)
                
                # Calculate scaling metrics
                if baseline_throughput is None:
                    baseline_throughput = avg_throughput
                    scaling_factor = 1.0
                    scaling_efficiency = 1.0
                else:
                    scaling_factor = avg_throughput / baseline_throughput
                    ideal_scaling_factor = config['gpu_count']
                    scaling_efficiency = scaling_factor / ideal_scaling_factor
                
                gpu_utilization = self._get_gpu_utilization()
                
                config_result = {
                    'gpu_count': config['gpu_count'],
                    'batch_size': config['batch_size'],
                    'memory_per_gpu_gb': config['memory_per_gpu'],
                    'total_memory_gb': config['memory_per_gpu'] * config['gpu_count'],
                    'average_throughput_patches_per_second': avg_throughput,
                    'throughput_std_dev': std_throughput,
                    'average_processing_time_seconds': avg_processing_time,
                    'scaling_factor': scaling_factor,
                    'scaling_efficiency': scaling_efficiency,
                    'gpu_utilization_percent': gpu_utilization,
                    'iterations_completed': len(throughputs)
                }
                
                results.append(config_result)
                
                logger.info(f"  Average throughput: {avg_throughput:.0f} patches/sec")
                logger.info(f"  Scaling efficiency: {scaling_efficiency:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to test {config['gpu_count']} GPU configuration: {e}")
                results.append({
                    'gpu_count': config['gpu_count'],
                    'error': str(e),
                    'scaling_efficiency': 0.0
                })
        
        # Calculate overall scaling metrics
        successful_results = [r for r in results if 'error' not in r]
        
        if successful_results:
            max_throughput = max(r['average_throughput_patches_per_second'] for r in successful_results)
            best_config = max(successful_results, key=lambda x: x['average_throughput_patches_per_second'])
            avg_scaling_efficiency = np.mean([r['scaling_efficiency'] for r in successful_results])
            
            # Check if scaling meets requirements (>70% efficiency for up to 4 GPUs)
            scaling_meets_requirements = all(
                r['scaling_efficiency'] >= 0.7 for r in successful_results if r['gpu_count'] <= 4
            )
            
            success_rate = len([r for r in successful_results if r['scaling_efficiency'] >= 0.7]) / len(successful_results)
        else:
            max_throughput = 0
            best_config = None
            avg_scaling_efficiency = 0
            scaling_meets_requirements = False
            success_rate = 0.0
        
        # Generate recommendations
        recommendations = []
        if not scaling_meets_requirements:
            recommendations.extend([
                "Optimize data loading pipeline for better GPU utilization",
                "Implement model parallelism for better multi-GPU scaling",
                "Consider using NCCL for efficient multi-GPU communication",
                "Profile GPU memory bandwidth utilization"
            ])
        
        if avg_scaling_efficiency < 0.8:
            recommendations.append("Investigate communication bottlenecks between GPUs")
        
        result = EnhancedPerformanceResult(
            test_name="Throughput Scaling",
            success_rate=success_rate,
            meets_requirements=scaling_meets_requirements,
            detailed_results=results,
            recommendations=recommendations,
            timestamp=time.time()
        )
        
        self.enhanced_results.append(result)
        
        logger.info(f"Throughput scaling benchmark completed:")
        logger.info(f"  Max throughput: {max_throughput:.0f} patches/sec")
        logger.info(f"  Average scaling efficiency: {avg_scaling_efficiency:.2f}")
        logger.info(f"  Scaling meets requirements: {scaling_meets_requirements}")
        
        return result
    
    async def _process_slide_with_memory_optimization(
        self, 
        slide_path: str, 
        config: StreamingConfig
    ) -> Dict[str, Any]:
        """Process slide with memory optimization and monitoring."""
        from .wsi_stream_reader import WSIStreamReader
        from .gpu_pipeline import GPUPipeline
        from .attention_aggregator import StreamingAttentionAggregator
        
        reader = WSIStreamReader(slide_path, config)
        gpu_pipeline = GPUPipeline(self.model, config)
        aggregator = StreamingAttentionAggregator(self.model, config)
        
        metadata = reader.initialize_streaming()
        
        patches_processed = 0
        batch_size_changes = 0
        oom_recoveries = 0
        start_time = time.time()
        
        try:
            async for tile_batch in reader.stream_tiles():
                try:
                    features = await gpu_pipeline.process_batch_async(tile_batch.tiles)
                    confidence_update = aggregator.update_features(features, tile_batch.coordinates)
                    
                    patches_processed += len(tile_batch.tiles)
                    
                    # Early stopping if confident enough
                    if confidence_update.early_stop_recommended:
                        break
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Handle OOM by reducing batch size
                        oom_recoveries += 1
                        batch_size_changes += 1
                        gpu_pipeline.optimize_batch_size(0.9)  # Reduce by 10%
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        except Exception as e:
            logger.error(f"Error processing slide: {e}")
            raise e
        
        processing_time = time.time() - start_time
        final_result = aggregator.finalize_prediction()
        
        return {
            'patches_processed': patches_processed,
            'processing_time': processing_time,
            'batch_size_changes': batch_size_changes,
            'oom_recoveries': oom_recoveries,
            'confidence': final_result.confidence,
            'peak_memory_gb': psutil.virtual_memory().used / (1024**3)
        }
    
    async def _process_slide_multi_gpu(
        self, 
        slide_path: str, 
        config: StreamingConfig
    ) -> Dict[str, Any]:
        """Process slide using multi-GPU configuration."""
        from .parallel_pipeline import ParallelGPUPipeline
        from .wsi_stream_reader import WSIStreamReader
        from .attention_aggregator import StreamingAttentionAggregator
        
        reader = WSIStreamReader(slide_path, config)
        # Use parallel GPU pipeline for multi-GPU processing
        gpu_pipeline = ParallelGPUPipeline(self.model, config)
        aggregator = StreamingAttentionAggregator(self.model, config)
        
        metadata = reader.initialize_streaming()
        
        patches_processed = 0
        start_time = time.time()
        
        async for tile_batch in reader.stream_tiles():
            features = await gpu_pipeline.process_batch_async(tile_batch.tiles)
            confidence_update = aggregator.update_features(features, tile_batch.coordinates)
            
            patches_processed += len(tile_batch.tiles)
            
            # Early stopping if confident enough
            if confidence_update.early_stop_recommended:
                break
        
        processing_time = time.time() - start_time
        final_result = aggregator.finalize_prediction()
        
        return {
            'patches_processed': patches_processed,
            'processing_time': processing_time,
            'confidence': final_result.confidence
        }
    
    def generate_enhanced_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance validation report."""
        if not self.enhanced_results:
            return {'error': 'No enhanced validation results available'}
        
        # Calculate overall metrics
        overall_success_rate = np.mean([r.success_rate for r in self.enhanced_results])
        overall_meets_requirements = all(r.meets_requirements for r in self.enhanced_results)
        
        # Collect all recommendations
        all_recommendations = []
        for result in self.enhanced_results:
            all_recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = list(dict.fromkeys(all_recommendations))
        
        report = {
            'enhanced_performance_validation_report': {
                'version': '1.0',
                'timestamp': time.time(),
                'overall_success_rate': overall_success_rate,
                'overall_meets_requirements': overall_meets_requirements,
                'test_results': [
                    {
                        'test_name': r.test_name,
                        'success_rate': r.success_rate,
                        'meets_requirements': r.meets_requirements,
                        'timestamp': r.timestamp
                    }
                    for r in self.enhanced_results
                ],
                'detailed_results': [r.detailed_results for r in self.enhanced_results],
                'recommendations': unique_recommendations,
                'performance_summary': {
                    'processing_time_validation': any(r.test_name == "30-Second Processing Requirement" and r.meets_requirements for r in self.enhanced_results),
                    'memory_usage_validation': any(r.test_name == "Memory Usage Bounds" and r.meets_requirements for r in self.enhanced_results),
                    'throughput_scaling_validation': any(r.test_name == "Throughput Scaling" and r.meets_requirements for r in self.enhanced_results)
                }
            }
        }
        
        return report


async def run_enhanced_performance_validation_suite(
    config: StreamingConfig,
    target_hardware: str = "RTX_4090"
) -> Dict[str, Any]:
    """
    Run comprehensive enhanced performance validation suite.
    
    This function orchestrates all enhanced performance validation tests:
    - Task 8.2.1.1: 30-second processing requirement validation
    - Task 8.2.1.2: Memory usage bounds testing
    - Task 8.2.1.3: Multi-GPU throughput scaling
    
    Args:
        config: Streaming configuration
        target_hardware: Target hardware specification
        
    Returns:
        Dict containing all enhanced validation results
    """
    validator = EnhancedPerformanceValidator(config)
    
    logger.info("Starting enhanced performance validation suite")
    
    # Task 8.2.1.1: Validate 30-second processing requirement
    logger.info("Running 30-second processing requirement validation...")
    time_validation = await validator.validate_30_second_processing_requirement(target_hardware)
    
    # Task 8.2.1.2: Test memory usage bounds
    logger.info("Running comprehensive memory usage bounds testing...")
    memory_validation = await validator.test_memory_usage_bounds_comprehensive()
    
    # Task 8.2.1.3: Benchmark throughput scaling
    logger.info("Running multi-GPU throughput scaling benchmark...")
    scaling_validation = await validator.benchmark_throughput_scaling_comprehensive()
    
    # Generate comprehensive validation report
    comprehensive_report = validator.generate_enhanced_performance_report()
    
    overall_validation_passed = (
        time_validation.meets_requirements and
        memory_validation.meets_requirements and
        scaling_validation.meets_requirements
    )
    
    validation_summary = {
        'enhanced_performance_validation_suite_version': '1.0',
        'validation_timestamp': time.time(),
        'target_hardware': target_hardware,
        'processing_time_validation': {
            'success_rate': time_validation.success_rate,
            'meets_requirements': time_validation.meets_requirements
        },
        'memory_usage_validation': {
            'success_rate': memory_validation.success_rate,
            'meets_requirements': memory_validation.meets_requirements
        },
        'throughput_scaling_validation': {
            'success_rate': scaling_validation.success_rate,
            'meets_requirements': scaling_validation.meets_requirements
        },
        'overall_validation_passed': overall_validation_passed,
        'comprehensive_report': comprehensive_report,
        'critical_requirements_status': {
            '30_second_processing': time_validation.meets_requirements,
            '2gb_memory_limit': memory_validation.meets_requirements,
            'multi_gpu_scaling': scaling_validation.meets_requirements
        }
    }
    
    logger.info("Enhanced performance validation suite completed")
    logger.info(f"Overall validation passed: {overall_validation_passed}")
    
    return validation_summary