#!/usr/bin/env python3
"""
Inference Pipeline Profiler

Profiles the current inference pipeline to identify bottlenecks and optimization opportunities.
"""

import time
import logging
import cProfile
import pstats
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from PIL import Image

from src.inference.inference_engine import InferenceEngine
from src.inference.model_loader import get_model_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceProfiler:
    """Profiles inference pipeline performance."""
    
    def __init__(self):
        self.engine = InferenceEngine()
        self.timings = {}
        
    def profile_stage(self, stage_name: str, func, *args, **kwargs):
        """Profile a single stage of the pipeline."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        self.timings[stage_name] = end_time - start_time
        logger.info(f"{stage_name}: {self.timings[stage_name]:.3f}s")
        return result
    
    def create_test_image(self, size=(224, 224)) -> Image.Image:
        """Create a test image for profiling."""
        # Create realistic pathology-like image
        np.random.seed(42)
        image_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        for i in range(0, size[0], 20):
            for j in range(0, size[1], 20):
                # Add some circular structures (cell-like)
                center_x, center_y = i + 10, j + 10
                for x in range(max(0, i), min(size[0], i + 20)):
                    for y in range(max(0, j), min(size[1], j + 20)):
                        dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                        if dist < 8:
                            image_array[x, y] = [200, 150, 200]  # Purple-ish
        
        return Image.fromarray(image_array)
    
    def profile_full_pipeline(self, num_runs: int = 5) -> Dict:
        """Profile the complete inference pipeline."""
        logger.info(f"Profiling inference pipeline with {num_runs} runs...")
        
        # Create test image
        test_image = self.create_test_image()
        test_path = "/tmp/test_pathology_image.png"
        test_image.save(test_path)
        
        all_timings = []
        
        for run in range(num_runs):
            logger.info(f"\n--- Run {run + 1}/{num_runs} ---")
            self.timings = {}
            
            # Profile complete pipeline
            total_start = time.perf_counter()
            
            # Stage 1: Image loading
            image = self.profile_stage("1_image_loading", self._load_image, test_path)
            
            # Stage 2: Model loading/preparation
            model_info = self.profile_stage("2_model_loading", self._get_model, "breast_cancer")
            
            # Stage 3: Preprocessing
            input_tensor = self.profile_stage("3_preprocessing", self._preprocess_image, 
                                            image, model_info['config'])
            
            # Stage 4: Model inference
            probabilities = self.profile_stage("4_model_inference", self._run_inference, 
                                             model_info['model'], input_tensor)
            
            # Stage 5: Postprocessing
            result = self.profile_stage("5_postprocessing", self._postprocess_results, 
                                      probabilities, model_info['config'])
            
            # Stage 6: Uncertainty calculation
            uncertainty = self.profile_stage("6_uncertainty", self._calculate_uncertainty, 
                                           probabilities)
            
            total_time = time.perf_counter() - total_start
            self.timings['total'] = total_time
            
            logger.info(f"Total time: {total_time:.3f}s")
            all_timings.append(self.timings.copy())
        
        # Calculate averages
        avg_timings = {}
        for stage in all_timings[0].keys():
            avg_timings[stage] = np.mean([t[stage] for t in all_timings])
            
        return {
            'average_timings': avg_timings,
            'all_runs': all_timings,
            'num_runs': num_runs
        }
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image (mimics InferenceEngine._load_image)."""
        return Image.open(image_path).convert('RGB')
    
    def _get_model(self, disease_type: str):
        """Get model info (mimics model loading)."""
        return self.engine.model_loader.get_model(disease_type)
    
    def _preprocess_image(self, image: Image.Image, config):
        """Preprocess image (mimics preprocessing)."""
        return self.engine.preprocessor.preprocess(image, config.preprocessing_config)
    
    def _run_inference(self, model, input_tensor):
        """Run model inference."""
        with torch.no_grad():
            logits = model(input_tensor)
            return torch.nn.functional.softmax(logits, dim=1)
    
    def _postprocess_results(self, probabilities, config):
        """Postprocess results."""
        return self.engine.postprocessor.process_results(
            probabilities=probabilities,
            class_names=config.class_names,
            model_name=config.name,
            model_version=config.version
        )
    
    def _calculate_uncertainty(self, probabilities):
        """Calculate uncertainty."""
        return self.engine._calculate_uncertainty(probabilities)
    
    def profile_with_cprofile(self, output_file: str = "inference_profile.prof"):
        """Profile using cProfile for detailed function-level analysis."""
        logger.info("Running detailed cProfile analysis...")
        
        test_image = self.create_test_image()
        test_path = "/tmp/test_pathology_image.png"
        test_image.save(test_path)
        
        def run_inference():
            return self.engine.analyze_image(test_path, "breast_cancer")
        
        # Run cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = run_inference()
        
        profiler.disable()
        profiler.dump_stats(output_file)
        
        # Print top functions
        stats = pstats.Stats(output_file)
        stats.sort_stats('cumulative')
        
        logger.info("\nTop 20 functions by cumulative time:")
        stats.print_stats(20)
        
        return result
    
    def analyze_bottlenecks(self, timings: Dict) -> Dict:
        """Analyze timing results to identify bottlenecks."""
        total_time = timings['total']
        
        analysis = {
            'total_time': total_time,
            'bottlenecks': [],
            'optimization_opportunities': []
        }
        
        # Identify stages taking >20% of total time
        for stage, time_val in timings.items():
            if stage == 'total':
                continue
                
            percentage = (time_val / total_time) * 100
            
            if percentage > 20:
                analysis['bottlenecks'].append({
                    'stage': stage,
                    'time': time_val,
                    'percentage': percentage
                })
        
        # Generate optimization recommendations
        if timings.get('1_image_loading', 0) > 1.0:
            analysis['optimization_opportunities'].append(
                "Image loading is slow - consider async loading or caching"
            )
        
        if timings.get('2_model_loading', 0) > 2.0:
            analysis['optimization_opportunities'].append(
                "Model loading is slow - implement model warming/caching"
            )
        
        if timings.get('3_preprocessing', 0) > 3.0:
            analysis['optimization_opportunities'].append(
                "Preprocessing is slow - consider GPU acceleration or batch processing"
            )
        
        if timings.get('4_model_inference', 0) > 5.0:
            analysis['optimization_opportunities'].append(
                "Model inference is slow - consider TensorRT, mixed precision, or model optimization"
            )
        
        if timings.get('5_postprocessing', 0) > 1.0:
            analysis['optimization_opportunities'].append(
                "Postprocessing is slow - streamline result formatting"
            )
        
        return analysis


def main():
    """Main profiling function."""
    logger.info("Starting inference pipeline profiling...")
    
    profiler = InferenceProfiler()
    
    try:
        # Profile pipeline stages
        results = profiler.profile_full_pipeline(num_runs=3)
        
        logger.info("\n" + "="*50)
        logger.info("PROFILING RESULTS")
        logger.info("="*50)
        
        avg_timings = results['average_timings']
        total_time = avg_timings['total']
        
        logger.info(f"Average total time: {total_time:.3f}s")
        logger.info(f"Target: <10s, Current: {'✓' if total_time < 10 else '✗'}")
        
        logger.info("\nStage breakdown:")
        for stage, time_val in avg_timings.items():
            if stage == 'total':
                continue
            percentage = (time_val / total_time) * 100
            logger.info(f"  {stage}: {time_val:.3f}s ({percentage:.1f}%)")
        
        # Analyze bottlenecks
        analysis = profiler.analyze_bottlenecks(avg_timings)
        
        if analysis['bottlenecks']:
            logger.info("\nBOTTLENECKS (>20% of total time):")
            for bottleneck in analysis['bottlenecks']:
                logger.info(f"  {bottleneck['stage']}: {bottleneck['time']:.3f}s ({bottleneck['percentage']:.1f}%)")
        
        if analysis['optimization_opportunities']:
            logger.info("\nOPTIMIZATION OPPORTUNITIES:")
            for i, opportunity in enumerate(analysis['optimization_opportunities'], 1):
                logger.info(f"  {i}. {opportunity}")
        
        # Run detailed cProfile
        logger.info("\nRunning detailed function-level profiling...")
        profiler.profile_with_cprofile("inference_profile.prof")
        
        logger.info(f"\nDetailed profile saved to: inference_profile.prof")
        logger.info("View with: python -m pstats inference_profile.prof")
        
    except Exception as e:
        logger.error(f"Profiling failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())